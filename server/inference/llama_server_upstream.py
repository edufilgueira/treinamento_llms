"""
Cliente HTTP para llama-server (llama.cpp): POST /v1/chat/completions.
Usado quando ORACULO_LLAMA_CPP_BASE_URL está definido — a UI Oráculo delega inferência ao binário C++.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Iterator
from urllib.parse import urljoin

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(600.0, connect=30.0)


def _base_url(raw: str) -> str:
    s = (raw or "").strip().rstrip("/")
    if not s:
        raise ValueError("base URL vazia")
    return s


def chat_template_kwargs_from_env() -> dict[str, Any] | None:
    raw = (os.environ.get("ORACULO_LLAMA_CPP_CHAT_TEMPLATE_KWARGS") or "").strip()
    if not raw:
        return None
    try:
        out = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"ORACULO_LLAMA_CPP_CHAT_TEMPLATE_KWARGS JSON inválido: {e}") from e
    return out if isinstance(out, dict) else None


def resolve_chat_template_kwargs_merged() -> dict[str, Any] | None:
    """Env ORACULO_LLAMA_CPP_CHAT_TEMPLATE_KWARGS + reasoning (off/on/auto) da base (admin)."""
    from server.db.auth_db import get_llama_server_settings

    base = chat_template_kwargs_from_env()
    out: dict[str, Any] = dict(base) if base else {}
    s = get_llama_server_settings()
    r = str(s.get("reasoning") or "off").strip().lower()
    if r == "off":
        out["enable_thinking"] = False
    elif r == "on":
        out["enable_thinking"] = True
    return out if out else None


def payload_sampling_extras_from_db() -> dict[str, Any]:
    """Campos extra para /v1/chat/completions (repeat_penalty, etc.) vindos da base."""
    from server.db.auth_db import get_llama_server_settings

    s = get_llama_server_settings()
    return {
        "repeat_penalty": float(s["repeat_penalty"]),
        "repeat_last_n": int(s["repeat_last_n"]),
        "reasoning_budget": int(s["reasoning_budget"]),
    }


def fetch_default_model_id(base: str, api_key: str | None) -> str:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = urljoin(base + "/", "v1/models")
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        r = client.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()
    rows = data.get("data") or []
    if rows and isinstance(rows, list):
        mid = rows[0].get("id")
        if mid:
            return str(mid)
    return "default"


def _headers_json(api_key: str | None) -> dict[str, str]:
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def _headers_sse(api_key: str | None) -> dict[str, str]:
    h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def build_payload(
    messages: list[dict],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stream: bool,
    chat_template_kwargs: dict[str, Any] | None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    p: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }
    if chat_template_kwargs:
        p["chat_template_kwargs"] = chat_template_kwargs
    if extra_fields:
        p.update(extra_fields)
    return p


def chat_completions_complete(
    base: str,
    messages: list[dict],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    api_key: str | None,
    chat_template_kwargs: dict[str, Any] | None,
    extra_fields: dict[str, Any] | None = None,
) -> tuple[str, dict[str, int] | None]:
    url = urljoin(_base_url(base) + "/", "v1/chat/completions")
    payload = build_payload(
        messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=False,
        chat_template_kwargs=chat_template_kwargs,
        extra_fields=extra_fields,
    )
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        r = client.post(url, json=payload, headers=_headers_json(api_key))
        r.raise_for_status()
        data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Resposta llama-server sem choices.")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        content = ""
    text = str(content)
    usage = data.get("usage")
    usage_out: dict[str, int] | None = None
    if isinstance(usage, dict):
        usage_out = {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
        }
    return text, usage_out


def chat_completions_stream_deltas(
    base: str,
    messages: list[dict],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    api_key: str | None,
    chat_template_kwargs: dict[str, Any] | None,
    cancel_event: threading.Event,
    extra_fields: dict[str, Any] | None = None,
) -> Iterator[str]:
    url = urljoin(_base_url(base) + "/", "v1/chat/completions")
    payload = build_payload(
        messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
        chat_template_kwargs=chat_template_kwargs,
        extra_fields=extra_fields,
    )
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        with client.stream("POST", url, json=payload, headers=_headers_sse(api_key)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if cancel_event.is_set():
                    break
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:].strip()
                elif line.startswith("data:"):
                    data = line[5:].strip()
                else:
                    continue
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in chunk:
                    err = chunk["error"]
                    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    raise RuntimeError(msg)
                chs = chunk.get("choices") or []
                if not chs:
                    continue
                delta = (chs[0].get("delta") or {}) if isinstance(chs[0], dict) else {}
                piece = delta.get("content")
                if piece:
                    yield str(piece)


def proxy_list_models(base: str, api_key: str | None) -> dict[str, Any]:
    url = urljoin(_base_url(base) + "/", "v1/models")
    with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
        r = client.get(url, headers=_headers_json(api_key))
        r.raise_for_status()
        return r.json()
