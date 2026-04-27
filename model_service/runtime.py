"""
Runtime único do modelo: HF (PyTorch), .gguf local (llama-cpp-python) **ou** llama-server HTTP.
Usado pelo servidor web e pela API /v1/chat/completions.
"""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_MS_ROOT = Path(__file__).resolve().parent.parent
for _p in (_MS_ROOT / "trein", _MS_ROOT / "server"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_runtime_singleton: "ModelRuntime | None" = None


class ModelRuntime:
    def __init__(self) -> None:
        self._tokenizer: Any = None
        self._model: Any = None
        self._llm: Any = None
        self._backend: str = "hf"
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
        if self._backend == "llama_server" and self._upstream_base:
            return True
        if self._backend == "gguf":
            return self._llm is not None
        return self._model is not None

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def model(self) -> Any:
        return self._model

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
        if self._backend == "llama_server":
            return max(0, len((text or "")) // 4)
        if self._backend == "gguf" and self._llm is not None:
            from .gguf_engine import count_output_tokens_gguf

            return count_output_tokens_gguf(self._llm, text)
        if self._tokenizer is None:
            return max(0, len((text or "")) // 4)
        return len(self._tokenizer.encode(str(text or ""), add_special_tokens=False))

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
        if self._backend == "llama_server" and self._upstream_base:
            out["llama_server_url"] = self._upstream_base
        return out

    def set_ui_only(self) -> None:
        self.ui_only = True
        self._tokenizer = None
        self._model = None
        self._llm = None
        self._upstream_base = None
        self._upstream_api_key = ""
        self._backend = "hf"
        self._mode = ""
        self._model_id = ""

    def load(
        self,
        model_name: str,
        adapter_dir: Path,
        merged: Path | None,
        *,
        base_only: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        from lora_engine import load_lora_pipeline

        self.ui_only = False
        self._llm = None
        self._upstream_base = None
        self._upstream_api_key = ""
        self._backend = "hf"
        tokenizer, model, merged_used = load_lora_pipeline(
            model_name,
            adapter_dir,
            merged,
            base_only=base_only,
            trust_remote_code=trust_remote_code,
            fix_generation_max_length=True,
        )
        self._tokenizer = tokenizer
        self._model = model
        if merged_used is not None:
            self._mode = "fundido"
        elif base_only:
            self._mode = "base"
        else:
            self._mode = "base+LoRA"
        self._model_id = str(merged_used.resolve()) if merged_used is not None else model_name

    def load_gguf(self, gguf_path: Path) -> None:
        from .gguf_engine import load_llama

        self.ui_only = False
        self._tokenizer = None
        self._model = None
        self._upstream_base = None
        self._upstream_api_key = ""
        self._llm = load_llama(gguf_path)
        self._backend = "gguf"
        self._mode = "gguf"
        self._model_id = str(gguf_path.resolve())

    def load_llama_server(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        from .llama_server_upstream import fetch_default_model_id, resolve_chat_template_kwargs_merged

        self.ui_only = False
        self._tokenizer = None
        self._model = None
        self._llm = None
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
        # Valida ORACULO_LLAMA_CPP_CHAT_TEMPLATE_KWARGS e lê reasoning da base.
        resolve_chat_template_kwargs_merged()

    def upstream_api_key(self) -> str | None:
        return self._upstream_api_key or None

    def clear(self) -> None:
        self._tokenizer = None
        self._model = None
        self._llm = None
        self._upstream_base = None
        self._upstream_api_key = ""
        self._backend = "hf"
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
            if self._backend == "llama_server":
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
                    max_tokens=max_new_tokens,
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
            if self._backend == "gguf":
                if self._llm is None:
                    raise RuntimeError("GGUF não carregado.")
                from .gguf_engine import generate_chat_reply_gguf

                text, usage = generate_chat_reply_gguf(
                    self._llm,
                    messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                if usage:
                    with self._last_openai_lock:
                        self._last_openai_usage = usage
                return text

            from lora_engine import generate_chat_reply

            if self._tokenizer is None or self._model is None:
                raise RuntimeError("Modelo não carregado.")
            return generate_chat_reply(
                self._tokenizer,
                self._model,
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

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
            if self._backend == "llama_server":
                if not self._upstream_base:
                    raise RuntimeError("llama-server não configurado.")
                with self._last_openai_lock:
                    self._last_openai_usage = None
                from .llama_server_upstream import (
                    chat_completions_stream_deltas,
                    payload_sampling_extras_from_db,
                    resolve_chat_template_kwargs_merged,
                )

                yield from chat_completions_stream_deltas(
                    self._upstream_base,
                    messages,
                    model=self._model_id,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    api_key=self.upstream_api_key(),
                    chat_template_kwargs=resolve_chat_template_kwargs_merged(),
                    cancel_event=cancel_event,
                    extra_fields=payload_sampling_extras_from_db(),
                )
                return
            if self._backend == "gguf":
                if self._llm is None:
                    raise RuntimeError("GGUF não carregado.")
                with self._last_openai_lock:
                    self._last_openai_usage = None
                from .gguf_engine import generate_chat_reply_stream_gguf

                yield from generate_chat_reply_stream_gguf(
                    self._llm,
                    messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    cancel_event=cancel_event,
                )
                return

            from lora_engine import generate_chat_reply_stream

            if self._tokenizer is None or self._model is None:
                raise RuntimeError("Modelo não carregado.")
            yield from generate_chat_reply_stream(
                self._tokenizer,
                self._model,
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                cancel_event=cancel_event,
            )


def get_runtime() -> ModelRuntime:
    global _runtime_singleton
    if _runtime_singleton is None:
        _runtime_singleton = ModelRuntime()
    return _runtime_singleton
