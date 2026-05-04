"""
Runtime único: inferência vía llama-server HTTP local/remoto ou Runpod Serverless.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator

_runtime_singleton: "ModelRuntime | None" = None


def inference_single_flight_enabled() -> bool:
    """
    Se True (omissão na BD), só uma geração corre de cada vez (fila global).
    Se False, vários utilizadores podem gerar em paralelo (ex.: backend Runpod com vários workers).
    Valor em tempo real: configuração global do admin (``app_global``).
    """
    return bool(get_runtime().inference_single_flight)


def cross_user_ui_block_enabled() -> bool:
    """
    Só quando ``inference_single_flight_enabled()`` é True.

    Se True (omissão na BD), ``GET /api/chat/generation-status`` indica quando *outro*
    utilizador está a gerar, para a UI mostrar o aviso e bloquear o envio.

    Se False, não há bloqueio/aviso na interface (os pedidos still serializam no servidor).
    """
    rt = get_runtime()
    if not rt.inference_single_flight:
        return False
    return bool(rt.ui_block_cross_user_generation)


class ModelRuntime:
    def __init__(self) -> None:
        self._backend: str = ""
        self._mode: str = ""
        self._model_id: str = ""
        self._upstream_base: str | None = None
        self._upstream_api_key: str = ""
        self._runpod_endpoint_id: str = ""
        self._runpod_api_key: str = ""
        self._runpod_poll_timeout_s: float = 900.0
        self._runpod_poll_interval_s: float = 1.0
        self.ui_only: bool = False
        self.inference_single_flight: bool = True
        self.ui_block_cross_user_generation: bool = True
        self._gen_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._active_uid: int | None = None
        self._active_gen_users: set[int] = set()
        self._last_openai_usage: dict[str, int] | None = None
        self._last_openai_lock = threading.Lock()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_loaded(self) -> bool:
        if self.ui_only:
            return False
        if self._backend == "runpod":
            return bool(self._runpod_endpoint_id and self._runpod_api_key)
        return self._backend == "llama_server" and bool(self._upstream_base)

    @property
    def tokenizer(self) -> Any:
        return None

    @property
    def model(self) -> Any:
        return None

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def upstream_base(self) -> str | None:
        return self._upstream_base

    @property
    def runpod_endpoint_id(self) -> str | None:
        return (self._runpod_endpoint_id or None) if self._backend == "runpod" else None

    def pop_openai_usage(self) -> dict[str, int] | None:
        with self._last_openai_lock:
            u = self._last_openai_usage
            self._last_openai_usage = None
            return u

    def count_output_tokens(self, text: str) -> int:
        return max(0, len((text or "")) // 4)

    def status_public(self) -> dict[str, Any]:
        if self.ui_only:
            return {"loaded": False, "ui_only": True}
        if not self.is_loaded:
            return {"loaded": False, "ui_only": False}
        out: dict[str, Any] = {
            "loaded": True,
            "ui_only": False,
            "mode": self._mode,
            "backend": self._backend,
            "model_name": self._model_id,
        }
        if self._upstream_base:
            out["llama_server_url"] = self._upstream_base
        if self._backend == "runpod" and self._runpod_endpoint_id:
            out["runpod_endpoint_id"] = self._runpod_endpoint_id
        out["inference_single_flight"] = inference_single_flight_enabled()
        out["ui_block_cross_user_generation"] = cross_user_ui_block_enabled()
        return out

    def apply_inference_queue_prefs(
        self,
        *,
        inference_single_flight: bool,
        ui_block_cross_user_generation: bool,
    ) -> None:
        self.inference_single_flight = bool(inference_single_flight)
        self.ui_block_cross_user_generation = bool(ui_block_cross_user_generation)

    def set_ui_only(self) -> None:
        self.ui_only = True
        self._upstream_base = None
        self._upstream_api_key = ""
        self._runpod_endpoint_id = ""
        self._runpod_api_key = ""
        self._runpod_poll_timeout_s = 900.0
        self._runpod_poll_interval_s = 1.0
        self._backend = ""
        self._mode = ""
        self._model_id = ""

    def load_llama_server(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        from .llama_server_upstream import fetch_default_model_id, resolve_chat_template_kwargs_merged

        self.ui_only = False
        self._runpod_endpoint_id = ""
        self._runpod_api_key = ""
        raw = (base_url or "").strip().rstrip("/")
        if not raw:
            raise ValueError("base_url do llama-server vazio.")
        self._upstream_base = raw
        self._upstream_api_key = (api_key or "").strip()
        key = self._upstream_api_key or None
        mid = (model or "").strip() or None
        if not mid:
            mid = fetch_default_model_id(self._upstream_base, key)
        self._model_id = mid
        self._backend = "llama_server"
        self._mode = "llama_server"
        resolve_chat_template_kwargs_merged()

    def load_runpod(
        self,
        endpoint_id: str,
        *,
        api_key: str | None,
        model_id: str | None = None,
        poll_timeout_s: float | None = None,
        poll_interval_s: float | None = None,
        startup_health: bool = False,
    ) -> None:
        """Delegação ao worker Serverless (handler compatível com messages/max_tokens/…)."""
        from .runpod_upstream import DEFAULT_POLL_INTERVAL_S, DEFAULT_POLL_TIMEOUT_S, runpod_endpoint_health

        self.ui_only = False
        self._upstream_base = None
        self._upstream_api_key = ""
        eid = (endpoint_id or "").strip()
        if not eid:
            raise ValueError("Runpod: endpoint id vazio.")
        key = (api_key or "").strip()
        if not key:
            raise ValueError("Runpod: API key obrigatória.")
        self._runpod_endpoint_id = eid
        self._runpod_api_key = key
        self._runpod_poll_timeout_s = float(
            poll_timeout_s if poll_timeout_s is not None and float(poll_timeout_s) > 0 else DEFAULT_POLL_TIMEOUT_S
        )
        self._runpod_poll_interval_s = float(
            poll_interval_s if poll_interval_s is not None and float(poll_interval_s) > 0 else DEFAULT_POLL_INTERVAL_S
        )
        self._model_id = (model_id or "").strip() or "runpod"
        self._backend = "runpod"
        self._mode = "runpod"
        if startup_health:
            runpod_endpoint_health(eid, key)

    def upstream_api_key(self) -> str | None:
        return self._upstream_api_key or None

    def clear(self) -> None:
        self._upstream_base = None
        self._upstream_api_key = ""
        self._runpod_endpoint_id = ""
        self._runpod_api_key = ""
        self._runpod_poll_timeout_s = 900.0
        self._runpod_poll_interval_s = 1.0
        self._backend = ""
        self._mode = ""
        self._model_id = ""
        with self._active_lock:
            self._active_uid = None
            self._active_gen_users.clear()
        with self._last_openai_lock:
            self._last_openai_usage = None

    @property
    def active_generation_user_ids(self) -> frozenset[int]:
        """Utilizadores com geração em curso (um com fila global; vários em modo concorrente)."""
        with self._active_lock:
            if inference_single_flight_enabled():
                if self._active_uid is not None:
                    return frozenset({self._active_uid})
                return frozenset()
            return frozenset(self._active_gen_users)

    @property
    def active_generation_user_id(self) -> int | None:
        with self._active_lock:
            if inference_single_flight_enabled():
                return self._active_uid
            return None

    @contextmanager
    def _generation_slot(self, user_id: int | None) -> Any:
        single = inference_single_flight_enabled()
        with self._active_lock:
            if single:
                self._active_uid = user_id
            elif user_id is not None:
                self._active_gen_users.add(int(user_id))
        try:
            if single:
                with self._gen_lock:
                    yield
            else:
                yield
        finally:
            with self._active_lock:
                if single:
                    self._active_uid = None
                elif user_id is not None:
                    self._active_gen_users.discard(int(user_id))

    def generate(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        user_id: int | None = None,
    ) -> str:
        if not self.is_loaded:
            raise RuntimeError("Modelo não carregado.")
        with self._generation_slot(user_id):
            from server_for_serveless.db.auth_db import get_llama_server_settings
            from server_for_serveless.services.llama_context import cap_max_new_tokens_for_n_ctx
            from server_for_serveless.services.history import finalize_messages_for_upstream

            ls = get_llama_server_settings()
            mnt_eff = cap_max_new_tokens_for_n_ctx(
                int(ls["n_ctx"]), int(max_new_tokens)
            )
            messages = finalize_messages_for_upstream(
                messages, user_id=user_id, max_new_tokens=mnt_eff
            )

            if self._backend == "runpod":
                from .runpod_upstream import runpod_chat_complete

                text, usage = runpod_chat_complete(
                    self._runpod_endpoint_id,
                    self._runpod_api_key,
                    messages=messages,
                    max_tokens=mnt_eff,
                    temperature=temperature,
                    top_p=top_p,
                    poll_timeout_s=self._runpod_poll_timeout_s,
                    poll_interval_s=self._runpod_poll_interval_s,
                )
            else:
                if not self._upstream_base:
                    raise RuntimeError("llama-server não configurado.")
                from .llama_server_upstream import (
                    chat_completions_complete,
                    payload_sampling_extras_from_db,
                    resolve_chat_template_kwargs_merged,
                )

                text, usage = chat_completions_complete(
                    self._upstream_base,
                    messages,
                    model=self._model_id,
                    max_tokens=mnt_eff,
                    temperature=temperature,
                    top_p=top_p,
                    api_key=self.upstream_api_key(),
                    chat_template_kwargs=resolve_chat_template_kwargs_merged(),
                    extra_fields=payload_sampling_extras_from_db(),
                )
            if usage:
                with self._last_openai_lock:
                    self._last_openai_usage = usage
            return text

    def stream(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        cancel_event: threading.Event,
        user_id: int | None = None,
    ) -> Iterator[str]:
        if not self.is_loaded:
            raise RuntimeError("Modelo não carregado.")
        with self._generation_slot(user_id):
            with self._last_openai_lock:
                self._last_openai_usage = None
            from server_for_serveless.db.auth_db import get_llama_server_settings
            from server_for_serveless.services.llama_context import cap_max_new_tokens_for_n_ctx
            from server_for_serveless.services.history import finalize_messages_for_upstream

            ls = get_llama_server_settings()
            mnt_eff = cap_max_new_tokens_for_n_ctx(
                int(ls["n_ctx"]), int(max_new_tokens)
            )
            messages = finalize_messages_for_upstream(
                messages, user_id=user_id, max_new_tokens=mnt_eff
            )

            if self._backend == "runpod":
                from .runpod_upstream import iter_runpod_chat_stream

                acc = ""
                for delta in iter_runpod_chat_stream(
                    self._runpod_endpoint_id,
                    self._runpod_api_key,
                    messages=messages,
                    max_tokens=mnt_eff,
                    temperature=temperature,
                    top_p=top_p,
                    cancel_event=cancel_event,
                    poll_timeout_s=self._runpod_poll_timeout_s,
                    poll_interval_s=self._runpod_poll_interval_s,
                ):
                    acc += delta
                    yield delta
                if acc:
                    with self._last_openai_lock:
                        self._last_openai_usage = {
                            "prompt_tokens": 0,
                            "completion_tokens": max(1, self.count_output_tokens(acc)),
                        }
                return

            if not self._upstream_base:
                raise RuntimeError("llama-server não configurado.")
            from .llama_server_upstream import (
                chat_completions_stream_deltas,
                payload_sampling_extras_from_db,
                resolve_chat_template_kwargs_merged,
            )

            yield from chat_completions_stream_deltas(
                self._upstream_base,
                messages,
                model=self._model_id,
                max_tokens=mnt_eff,
                temperature=temperature,
                top_p=top_p,
                api_key=self.upstream_api_key(),
                chat_template_kwargs=resolve_chat_template_kwargs_merged(),
                cancel_event=cancel_event,
                extra_fields=payload_sampling_extras_from_db(),
            )


def get_runtime() -> ModelRuntime:
    global _runtime_singleton
    if _runtime_singleton is None:
        _runtime_singleton = ModelRuntime()
    return _runtime_singleton
