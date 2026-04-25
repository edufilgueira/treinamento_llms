"""
Inferência com ficheiro .gguf via llama-cpp-python (alinhado a chat, sem PyTorch/Transformers).

Mensagens no formato OpenAI/Oráculo; o modelo aplica o *chat template* embebido no GGUF.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw, 10)
    except ValueError:
        return default


def load_llama(
    gguf_path: Path,
    *,
    n_ctx: int | None = None,
    n_gpu_layers: int | None = None,
) -> Any:
    try:
        from llama_cpp import Llama
    except ImportError as err:
        raise RuntimeError(
            "Falta o pacote llama-cpp-python. Instale, por ex.:  "
            "pip install -r server/requirements-gguf.txt"
        ) from err

    n_ctx = n_ctx if n_ctx is not None else _int_env("ORACULO_GGUF_N_CTX", 4096)
    n_gpu = n_gpu_layers if n_gpu_layers is not None else _int_env("ORACULO_GGUF_N_GPU_LAYERS", -1)

    return Llama(
        model_path=str(gguf_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu,
        verbose=False,
    )


def _messages_for_llama(messages: list[dict]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        role = str(m.get("role", "user") or "user")
        if role not in ("system", "user", "assistant"):
            role = "user"
        out.append({"role": role, "content": str(m.get("content", ""))})
    return out


def generate_chat_reply_gguf(
    llm: Any,
    messages: list[dict],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> tuple[str, dict[str, int] | None]:
    response = llm.create_chat_completion(
        messages=_messages_for_llama(messages),
        max_tokens=max_new_tokens,
        temperature=float(max(0.0, temperature)),
        top_p=float(max(0.0, min(1.0, top_p))),
        stream=False,
    )
    usage = response.get("usage")
    u_out: dict[str, int] | None = None
    if isinstance(usage, dict):
        u_out = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
    choice = (response.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = (msg.get("content") or "").strip()
    return text, u_out


def generate_chat_reply_stream_gguf(
    llm: Any,
    messages: list[dict],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    cancel_event: threading.Event | None = None,
) -> Iterator[str]:
    stream = llm.create_chat_completion(
        messages=_messages_for_llama(messages),
        max_tokens=max_new_tokens,
        temperature=float(max(0.0, temperature)),
        top_p=float(max(0.0, min(1.0, top_p))),
        stream=True,
    )
    for chunk in stream:
        if cancel_event is not None and cancel_event.is_set():
            break
        try:
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = (choices[0].get("delta") or {}) if isinstance(choices[0], dict) else {}
            content = delta.get("content")
        except (KeyError, IndexError, TypeError):
            continue
        if content:
            yield content


def count_output_tokens_gguf(llm: Any, text: str) -> int:
    if not (text or "").strip():
        return 0
    data = text.encode("utf-8", errors="replace")
    toks = llm.tokenize(data, add_bos=False, special=False)
    return len(toks) if toks is not None else 0
