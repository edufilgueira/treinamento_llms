"""
Cliente HTTP para Runpod Serverless (endpoint em fila).

Envia o mesmo ``input`` que o ``handler.py`` do worker espera (messages, max_tokens, …).
Respostas **sem** streaming usam ``POST /run`` + ``GET /status``. Respostas **com**
streaming (UI token a token) usam ``POST /run`` + ``GET /stream/{job_id}`` — o worker
deve ser um **streaming handler** que faz yield dos deltas (ver ``handler.py`` na raiz).

Referência: https://docs.runpod.io/serverless/endpoints/operation-reference
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Iterator

import httpx

API_BASE = "https://api.runpod.ai/v2"
DEFAULT_POLL_INTERVAL_S = 1.0
DEFAULT_POLL_TIMEOUT_S = 900.0


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _piece_to_text(p: Any) -> str:
    """Um único segmento agregado (``return_aggregate_stream`` pode ser str ou dict)."""
    if p is None:
        return ""
    if isinstance(p, str):
        return p
    if isinstance(p, (int, float, bool)):
        return str(p)
    if isinstance(p, dict):
        err = p.get("error")
        if err:
            raise RuntimeError(str(err))
        for k in ("text", "content"):
            v = p.get(k)
            if isinstance(v, str):
                return v
        inner = p.get("output")
        if isinstance(inner, str):
            return inner
        if isinstance(inner, list):
            return "".join(_piece_to_text(x) for x in inner)
        return ""
    if isinstance(p, list):
        return "".join(_piece_to_text(x) for x in p)
    return str(p)


def _parse_worker_output(raw: Any) -> str:
    """Interpreta o retorno do handler Runpod (string, lista agregada, ou dict)."""
    if raw is None:
        raise RuntimeError("Runpod devolveu output vazio.")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        if len(raw) == 0:
            raise RuntimeError(
                "Runpod completou com lista de saída vazia (0 yields do handler). "
                "Confirme no painel Runpod os logs do worker: llama-server a responder, "
                "handler em streaming e mesma imagem que `handler.py` do repositório."
            )
        return "".join(_piece_to_text(x) for x in raw)
    if isinstance(raw, dict):
        err = raw.get("error")
        if err:
            raise RuntimeError(str(err))
        out = raw.get("output")
        if out is None:
            raise RuntimeError(f"Runpod output sem campo 'output': {raw!r}")
        if isinstance(out, str):
            return out
        if isinstance(out, list):
            if len(out) == 0:
                raise RuntimeError(
                    "Runpod output.output é uma lista vazia (handler não devolveu texto). "
                    "Veja logs do worker e se `/v1/chat/completions` com stream devolve deltas `content`."
                )
            return "".join(_piece_to_text(x) for x in out)
        if isinstance(out, dict) and "output" in out:
            inner = out.get("output")
            if isinstance(inner, str):
                return inner
            if isinstance(inner, list):
                return "".join(_piece_to_text(x) for x in inner)
        return _piece_to_text(out) or str(out)
    return _piece_to_text(raw) or str(raw)


def _extract_stream_deltas(obj: Any) -> list[str]:
    """Interpreta um chunk JSON da API Runpod ``/stream`` (lista ou dict com ``output``)."""
    out: list[str] = []
    if obj is None:
        return out
    if isinstance(obj, str):
        if obj:
            out.append(obj)
        return out
    if isinstance(obj, list):
        for el in obj:
            out.extend(_extract_stream_deltas(el))
        return out
    if not isinstance(obj, dict):
        return out
    err = obj.get("error")
    if err:
        raise RuntimeError(str(err))
    payload = obj.get("output")
    if isinstance(payload, str) and payload:
        out.append(payload)
        return out
    if isinstance(payload, dict):
        t = payload.get("text")
        if isinstance(t, list):
            for s in t:
                if s:
                    out.append(str(s))
            return out
        if isinstance(t, str) and t:
            out.append(t)
            return out
        d = payload.get("delta")
        if isinstance(d, dict):
            piece = d.get("content")
            if isinstance(piece, str) and piece:
                out.append(piece)
                return out
        elif isinstance(d, str) and d:
            out.append(d)
            return out
        c = payload.get("content")
        if isinstance(c, str) and c:
            out.append(c)
            return out
        return out
    if isinstance(payload, list):
        for s in payload:
            if isinstance(s, str) and s:
                out.append(s)
            else:
                out.extend(_extract_stream_deltas(s))
        return out
    return out


def _poll_runpod_job_output(
    client: httpx.Client,
    endpoint_id: str,
    api_key: str,
    job_id: str,
    *,
    poll_interval_s: float,
    timeout_s: float,
) -> tuple[str, dict[str, int] | None]:
    """Espera por ``COMPLETED`` e devolve texto + usage (``GET /status``)."""
    eid = (endpoint_id or "").strip()
    key = (api_key or "").strip()
    status_url = f"{API_BASE}/{eid}/status/{job_id}"
    deadline = time.monotonic() + max(15.0, float(timeout_s))
    interval = max(0.2, float(poll_interval_s))

    while time.monotonic() < deadline:
        time.sleep(interval)
        sr = client.get(status_url, headers=_headers(key))
        sr.raise_for_status()
        st = sr.json()
        status = (st.get("status") or "").upper()
        if status == "COMPLETED":
            text = _parse_worker_output(st.get("output"))
            usage_out: dict[str, int] | None = None
            u = st.get("usage") if isinstance(st.get("usage"), dict) else None
            if isinstance(u, dict):
                usage_out = {
                    "prompt_tokens": int(u.get("input_tokens") or u.get("prompt_tokens") or 0),
                    "completion_tokens": int(
                        u.get("output_tokens") or u.get("completion_tokens") or 0
                    ),
                }
            return text, usage_out
        if status in ("FAILED", "CANCELLED", "TIMED_OUT", "ERROR"):
            err = st.get("error") or st
            raise RuntimeError(f"Runpod job {status}: {err}")

    raise TimeoutError(
        f"Timeout ({timeout_s}s) à espera do job Runpod {job_id!r}. "
        "Aumenta o tempo máx. nas configurações Runpod (admin) ou ORACULO_RUNPOD_POLL_TIMEOUT_S."
    )


def _cancel_runpod_job(client: httpx.Client, eid: str, api_key: str, job_id: str) -> None:
    try:
        url = f"{API_BASE}/{eid}/cancel/{job_id}"
        client.post(url, headers=_headers(api_key), timeout=30.0)
    except Exception:
        pass


def iter_runpod_chat_stream(
    endpoint_id: str,
    api_key: str,
    *,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    cancel_event: threading.Event | None = None,
    poll_timeout_s: float | None = None,
    poll_interval_s: float | None = None,
) -> Iterator[str]:
    """
    ``POST /run`` + ``GET /stream/{job_id}`` para deltas.

    Se o `/stream` não devolver texto parseável (formato SSE/JSON diferente), faz
    fall-back para o mesmo job com ``GET /status`` (saída agregada), como no chat
    não-stream — evita resposta vazia e 0 tokens na UI.
    """
    eid = (endpoint_id or "").strip()
    if not eid:
        raise ValueError("endpoint_id Runpod vazio.")
    key = (api_key or "").strip()
    if not key:
        raise ValueError("API key Runpod vazia.")

    input_obj: dict[str, Any] = {
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }

    read_timeout = float(
        poll_timeout_s
        if poll_timeout_s is not None and float(poll_timeout_s) > 0
        else (os.environ.get("ORACULO_RUNPOD_POLL_TIMEOUT_S") or "").strip()
        or DEFAULT_POLL_TIMEOUT_S
    )
    interval = float(
        poll_interval_s
        if poll_interval_s is not None and float(poll_interval_s) > 0
        else (os.environ.get("ORACULO_RUNPOD_POLL_INTERVAL_S") or "").strip()
        or DEFAULT_POLL_INTERVAL_S
    )
    run_url = f"{API_BASE}/{eid}/run"
    headers_stream = {**_headers(key), "Accept": "application/json"}

    timeout_cfg = httpx.Timeout(connect=30.0, read=max(60.0, read_timeout), write=120.0, pool=30.0)

    with httpx.Client(timeout=timeout_cfg) as client:
        r = client.post(run_url, headers=_headers(key), json={"input": input_obj})
        r.raise_for_status()
        job = r.json()
        job_id = job.get("id")
        if not job_id:
            raise RuntimeError(f"Runpod /run sem id: {job!r}")

        stream_url = f"{API_BASE}/{eid}/stream/{job_id}"
        stream_had_delta = False
        with client.stream("GET", stream_url, headers=headers_stream) as resp:
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as err:
                if err.response.status_code in (400, 404, 405):
                    raise RuntimeError(
                        "Runpod /stream falhou — atualiza a imagem do worker para o handler "
                        "em streaming (`handler.py` com yield + `return_aggregate_stream`)."
                    ) from err
                raise

            buffer = ""
            stream_stopped = False
            for chunk in resp.iter_text():
                if stream_stopped:
                    break
                if cancel_event is not None and cancel_event.is_set():
                    _cancel_runpod_job(client, eid, key, job_id)
                    break
                if not chunk:
                    continue
                buffer += chunk
                while "\n" in buffer and not stream_stopped:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        stream_stopped = True
                        break
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for delta in _extract_stream_deltas(obj):
                        if delta:
                            stream_had_delta = True
                            yield delta

            if cancel_event is not None and cancel_event.is_set():
                return
            tail = buffer.strip()
            if tail:
                if tail.startswith("data:"):
                    tail = tail[5:].strip()
                try:
                    obj = json.loads(tail)
                    for delta in _extract_stream_deltas(obj):
                        if delta:
                            stream_had_delta = True
                            yield delta
                except json.JSONDecodeError:
                    pass

        if cancel_event is not None and cancel_event.is_set():
            return
        if not stream_had_delta:
            text, _u = _poll_runpod_job_output(
                client,
                eid,
                key,
                job_id,
                poll_interval_s=max(0.2, interval),
                timeout_s=max(30.0, read_timeout),
            )
            if text.strip():
                yield text


def runpod_chat_complete(
    endpoint_id: str,
    api_key: str,
    *,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    poll_timeout_s: float | None = None,
    poll_interval_s: float | None = None,
) -> tuple[str, dict[str, int] | None]:
    """
    Executa uma geração vía Runpod e devolve (texto, usage|None).
    """
    eid = (endpoint_id or "").strip()
    if not eid:
        raise ValueError("endpoint_id Runpod vazio.")
    key = (api_key or "").strip()
    if not key:
        raise ValueError("API key Runpod vazia.")

    input_obj: dict[str, Any] = {
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }

    timeout_s = float(
        poll_timeout_s
        if poll_timeout_s is not None and float(poll_timeout_s) > 0
        else (os.environ.get("ORACULO_RUNPOD_POLL_TIMEOUT_S") or "").strip()
        or DEFAULT_POLL_TIMEOUT_S
    )
    interval = float(
        poll_interval_s
        if poll_interval_s is not None and float(poll_interval_s) > 0
        else (os.environ.get("ORACULO_RUNPOD_POLL_INTERVAL_S") or "").strip()
        or DEFAULT_POLL_INTERVAL_S
    )

    run_url = f"{API_BASE}/{eid}/run"

    with httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0)) as client:
        r = client.post(
            run_url,
            headers=_headers(key),
            json={"input": input_obj},
        )
        r.raise_for_status()
        job = r.json()
        job_id = job.get("id")
        if not job_id:
            raise RuntimeError(f"Runpod /run sem id: {job!r}")

        return _poll_runpod_job_output(
            client,
            eid,
            key,
            job_id,
            poll_interval_s=interval,
            timeout_s=timeout_s,
        )


def runpod_endpoint_health(endpoint_id: str, api_key: str) -> dict[str, Any]:
    """Verificação rápida no arranque (opcional)."""
    eid = (endpoint_id or "").strip()
    key = (api_key or "").strip()
    url = f"{API_BASE}/{eid}/health"
    with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        r = client.get(url, headers=_headers(key))
        r.raise_for_status()
        return r.json()
