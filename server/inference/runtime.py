"""
Runtime único: delegação ao llama-server (llama.cpp) via HTTP.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator

_runtime_singleton: "ModelRuntime | None" = None


class ModelRuntime:
    def __init__(self) -> None:
        self._backend: str = ""
        self._mode: str = ""
        self._model_id: str = ""
        self._upstream_base: str | None = None
        self._upstream_api_key: str = ""
        self.ui_only: bool = False
        self._gen_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._active_uid: int | None = None
        self._last_openai_usage: dict[str, int] | None = None
        self._last_openai_lock = threading.Lock()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_loaded(self) -> bool:
        if self.ui_only:
            return False
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
        return out

    def set_ui_only(self) -> None:
        self.ui_only = True
        self._upstream_base = None
        self._upstream_api_key = ""
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

    def upstream_api_key(self) -> str | None:
        return self._upstream_api_key or None

    def clear(self) -> None:
        self._upstream_base = None
        self._upstream_api_key = ""
        self._backend = ""
        self._mode = ""
        self._model_id = ""
        with self._last_openai_lock:
            self._last_openai_usage = None

    @property
    def active_generation_user_id(self) -> int | None:
        with self._active_lock:
            return self._active_uid

    @contextmanager
    def _generation_slot(self, user_id: int | None) -> Any:
        with self._active_lock:
            self._active_uid = user_id
        try:
            with self._gen_lock:
                yield
        finally:
            with self._active_lock:
                self._active_uid = None

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
            if not self._upstream_base:
                raise RuntimeError("llama-server não configurado.")
            from server.db.auth_db import get_llama_server_settings
            from server.services.llama_context import cap_max_new_tokens_for_n_ctx
            from server.services.history import finalize_messages_for_upstream

            from .llama_server_upstream import (
                chat_completions_complete,
                payload_sampling_extras_from_db,
                resolve_chat_template_kwargs_merged,
            )

            ls = get_llama_server_settings()
            mnt_eff = cap_max_new_tokens_for_n_ctx(
                int(ls["n_ctx"]), int(max_new_tokens)
            )
            messages = finalize_messages_for_upstream(
                messages, user_id=user_id, max_new_tokens=mnt_eff
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
            if not self._upstream_base:
                raise RuntimeError("llama-server não configurado.")
            with self._last_openai_lock:
                self._last_openai_usage = None
            from server.db.auth_db import get_llama_server_settings
            from server.services.llama_context import cap_max_new_tokens_for_n_ctx
            from server.services.history import finalize_messages_for_upstream

            from .llama_server_upstream import (
                chat_completions_stream_deltas,
                payload_sampling_extras_from_db,
                resolve_chat_template_kwargs_merged,
            )

            ls = get_llama_server_settings()
            mnt_eff = cap_max_new_tokens_for_n_ctx(
                int(ls["n_ctx"]), int(max_new_tokens)
            )
            messages = finalize_messages_for_upstream(
                messages, user_id=user_id, max_new_tokens=mnt_eff
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
