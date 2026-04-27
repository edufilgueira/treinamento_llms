"""
Inferência com ficheiro .gguf via llama-cpp-python (alinhado a chat, sem PyTorch/Transformers).

Mensagens no formato OpenAI/Oráculo; o modelo aplica o *chat template* embebido no GGUF.
"""

from __future__ import annotations

import os
import re
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


# Qwen3 (e similares) podem emitir raciocínio entre marcas; o template HF usa enable_thinking=False,
# mas o GGUF (llama.cpp) precisa de chat_template_kwargs e/ou pós-processamento.
_THINKING_BLOCK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<\|think\|>.*?<\|/think\|>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
)


def _strip_thinking_blocks(text: str) -> str:
    if not text:
        return text
    s = text
    for _ in range(8):
        prev = s
        for pat in _THINKING_BLOCK_PATTERNS:
            s = pat.sub("", s)
        if s == prev:
            break
    return s.strip()


_RE_THINKING_OPEN = re.compile(
    r"(?:<\|think\|>|<think>)",
    re.IGNORECASE,
)
_RE_THINKING_CLOSE = re.compile(
    r"(?:<\|/think\|>|</think>)",
    re.IGNORECASE,
)


def _stream_skip_leading_thinking(stream: Iterator[str]) -> Iterator[str]:
    """Remove um bloco inicial de pensamento (streaming) e reencaminha o resto."""
    buf: list[str] = []
    it = iter(stream)
    max_hold = 262_144
    early_plain = 2048

    for chunk in it:
        buf.append(chunk)
        s = "".join(buf)
        if _RE_THINKING_OPEN.search(s):
            m = _RE_THINKING_CLOSE.search(s)
            if m:
                tail = s[m.end() :]
                if tail:
                    yield tail
                for c in it:
                    yield c
                return
            if len(s) > max_hold:
                yield _strip_thinking_blocks(s)
                for c in it:
                    yield c
                return
        elif len(s) >= early_plain:
            yield s
            for c in it:
                yield c
            return

    if buf:
        yield _strip_thinking_blocks("".join(buf))


def _qwen_chat_template_extra() -> dict[str, Any]:
    return {"chat_template_kwargs": {"enable_thinking": False}}


def _create_chat_completion(llm: Any, **kwargs: Any) -> Any:
    try:
        return llm.create_chat_completion(**kwargs, **_qwen_chat_template_extra())
    except TypeError:
        return llm.create_chat_completion(**kwargs)


def _gguf_jinja_chat_prompt(
    llm: Any,
    messages: list[dict[str, str]],
) -> Any | None:
    """
    Aplica o chat template Jinja embutido no GGUF com ``enable_thinking: false``.

    O ``create_chat_completion`` do llama-cpp-python **não** encaminha
    ``chat_template_kwargs`` ao formatter (só lista fixa de argumentos), por isso
    o caminho fiável para Qwen3 é renderizar aqui e usar ``create_completion``.
    """
    try:
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    except ImportError:
        return None
    md = getattr(llm, "metadata", None) or {}
    template = md.get("tokenizer.chat_template")
    if not template:
        return None
    eos_id = int(llm.token_eos())
    bos_id = int(llm.token_bos())
    try:
        eos_txt = llm.detokenize([eos_id], special=True).decode("utf-8", errors="replace")
        bos_txt = llm.detokenize([bos_id], special=True).decode("utf-8", errors="replace")
    except Exception:
        eos_txt, bos_txt = "", ""
    stop_ids = [eos_id] if eos_id >= 0 else []
    fmt = Jinja2ChatFormatter(
        template=template,
        eos_token=eos_txt,
        bos_token=bos_txt,
        add_generation_prompt=True,
        stop_token_ids=stop_ids if stop_ids else None,
    )
    extra: dict[str, Any] = {"enable_thinking": False, "reasoning_budget": 0}
    try:
        return fmt(messages=messages, **extra)
    except Exception:
        try:
            return fmt(messages=messages, enable_thinking=False)
        except Exception:
            return None


def _stops_for_completion(formatted: Any) -> list[str]:
    raw = getattr(formatted, "stop", None)
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, list):
        return [s for s in raw if s]
    return []


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

    common = dict(
        model_path=str(gguf_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu,
        verbose=False,
    )
    try:
        return Llama(**common, **_qwen_chat_template_extra())
    except TypeError:
        pass
    return Llama(**common)


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
    msgs = _messages_for_llama(messages)
    formatted = _gguf_jinja_chat_prompt(llm, msgs)

    if formatted is not None:
        response = llm.create_completion(
            prompt=formatted.prompt,
            max_tokens=max_new_tokens,
            temperature=float(max(0.0, temperature)),
            top_p=float(max(0.0, min(1.0, top_p))),
            stream=False,
            stop=_stops_for_completion(formatted),
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
        text = _strip_thinking_blocks((choice.get("text") or "").strip())
        return text, u_out

    response = _create_chat_completion(
        llm,
        messages=msgs,
        max_tokens=max_new_tokens,
        temperature=float(max(0.0, temperature)),
        top_p=float(max(0.0, min(1.0, top_p))),
        stream=False,
    )
    usage = response.get("usage")
    u_out = None
    if isinstance(usage, dict):
        u_out = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
    choice = (response.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = _strip_thinking_blocks((msg.get("content") or "").strip())
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
    msgs = _messages_for_llama(messages)
    formatted = _gguf_jinja_chat_prompt(llm, msgs)

    if formatted is not None:
        stream = llm.create_completion(
            prompt=formatted.prompt,
            max_tokens=max_new_tokens,
            temperature=float(max(0.0, temperature)),
            top_p=float(max(0.0, min(1.0, top_p))),
            stream=True,
            stop=_stops_for_completion(formatted),
        )

        def _completion_stream_texts() -> Iterator[str]:
            for chunk in stream:
                if cancel_event is not None and cancel_event.is_set():
                    break
                try:
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    piece = choices[0].get("text") if isinstance(choices[0], dict) else None
                except (KeyError, IndexError, TypeError):
                    continue
                if piece:
                    yield piece

        yield from _completion_stream_texts()
        return

    stream = _create_chat_completion(
        llm,
        messages=msgs,
        max_tokens=max_new_tokens,
        temperature=float(max(0.0, temperature)),
        top_p=float(max(0.0, min(1.0, top_p))),
        stream=True,
    )

    def _delta_texts() -> Iterator[str]:
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

    yield from _stream_skip_leading_thinking(_delta_texts())


def count_output_tokens_gguf(llm: Any, text: str) -> int:
    if not (text or "").strip():
        return 0
    data = text.encode("utf-8", errors="replace")
    toks = llm.tokenize(data, add_bos=False, special=False)
    return len(toks) if toks is not None else 0
