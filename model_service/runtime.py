"""
Runtime único do modelo: carrega tokenizer + rede (HF) **ou** Llama .gguf e serializa geração.
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

    def pop_openai_usage(self) -> dict[str, int] | None:
        with self._last_openai_lock:
            u = self._last_openai_usage
            self._last_openai_usage = None
            return u

    def count_output_tokens(self, text: str) -> int:
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
        return {
            "loaded": True,
            "ui_only": False,
            "mode": self._mode,
            "backend": self._backend,
            "model_name": self._model_id,
        }

    def set_ui_only(self) -> None:
        self.ui_only = True
        self._tokenizer = None
        self._model = None
        self._llm = None
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
        self._llm = load_llama(gguf_path)
        self._backend = "gguf"
        self._mode = "gguf"
        self._model_id = str(gguf_path.resolve())

    def clear(self) -> None:
        self._tokenizer = None
        self._model = None
        self._llm = None
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
