#!/usr/bin/env python3
"""
Servidor local (FastAPI): carrega base + LoRA **uma vez** e serve chat por HTTP.

Evita repetir o custo de carregar vários GB em cada execução de inferir.py.

  cd <raiz do projeto> && python3 server/serve_lora.py
  # ou: ./server/serve.sh

Por defeito escuta em 0.0.0.0 (todas as interfaces): aceda com http://IP-DO-SERVIDOR:8765/
Nesta máquina: http://127.0.0.1:8765/ — só local: --host 127.0.0.1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

_SERVER_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVER_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_config import (
    DEFAULT_ADAPTER_DIR,
    DEFAULT_MERGED_MODEL_DIR,
    DEFAULT_MODEL_NAME,
    apply_loading_progress_env,
)

apply_loading_progress_env()

if not os.environ.get("PYTORCH_ALLOC_CONF"):
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from lora_engine import (
    generate_chat_reply,
    generate_chat_reply_stream,
    load_lora_pipeline,
)

STATIC_DIR = _SERVER_DIR / "static"

_gen_lock = threading.Lock()
_engine: dict | None = None
_startup_args: argparse.Namespace | None = None

# Geração desacoplada: continua ainda que o cliente (telemóvel) corte a ligação; leitura por GET.
_jobs_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}
_MAX_JOBS_BUFFER = 64


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Servidor local para o modelo LoRA treinado.")
    p.add_argument(
        "--host",
        default="0.0.0.0",
        help="Endereço a escutar. 0.0.0.0 = todas as interfaces (outras máquinas na rede). "
        "127.0.0.1 = só este computador.",
    )
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    p.add_argument(
        "--adapter_dir",
        type=Path,
        default=DEFAULT_ADAPTER_DIR,
        help="Pasta do adapter (train_lora.py). Ignorado se --merged_model_dir válido.",
    )
    p.add_argument(
        "--merged_model_dir",
        type=Path,
        default=None,
        help=f"Se existir com config.json, carrega só o modelo fundido (merge_lora.py). "
        f"Padrão: tentar {DEFAULT_MERGED_MODEL_DIR} se existir.",
    )
    p.add_argument("--trust_remote_code", action="store_true")
    return p.parse_args()


def _resolve_merged_path(cli_path: Path | None) -> Path | None:
    if cli_path is not None:
        return cli_path
    if DEFAULT_MERGED_MODEL_DIR.is_dir() and (DEFAULT_MERGED_MODEL_DIR / "config.json").is_file():
        return DEFAULT_MERGED_MODEL_DIR
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    args = _startup_args if _startup_args is not None else _parse_args()
    merged = _resolve_merged_path(args.merged_model_dir)
    print("A carregar modelo (só uma vez; pode demorar)…", flush=True)
    try:
        tokenizer, model = load_lora_pipeline(
            args.model_name,
            args.adapter_dir,
            merged,
            trust_remote_code=args.trust_remote_code,
            fix_generation_max_length=True,
        )
    except Exception as err:
        print(f"Erro ao carregar: {err}", file=sys.stderr, flush=True)
        raise
    mode = "fundido" if merged else "base+LoRA"
    if args.host in ("0.0.0.0", "::", "[::]"):
        print(
            f"Pronto ({mode}). À escuta em {args.host}:{args.port} — nesta máquina: "
            f"http://127.0.0.1:{args.port}/ — noutro PC/rede: http://<IP>:{args.port}/",
            flush=True,
        )
    else:
        print(f"Pronto ({mode}). Servidor em http://{args.host}:{args.port}/", flush=True)
    _engine = {
        "tokenizer": tokenizer,
        "model": model,
        "mode": mode,
        "model_name": args.model_name,
    }
    yield
    _engine = None


app = FastAPI(title="Oráculo LoRA local", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    max_new_tokens: int = Field(2048, ge=16, le=4096)
    temperature: float = Field(0.7, ge=0.01, le=2.0)
    top_p: float = Field(0.9, ge=0.05, le=1.0)


class ChatOut(BaseModel):
    reply: str


class JobCreateOut(BaseModel):
    job_id: str


class JobStateOut(BaseModel):
    status: str
    text: str
    error: str | None = None


def _run_generate(
    messages: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    assert _engine is not None
    tokenizer = _engine["tokenizer"]
    model = _engine["model"]
    return generate_chat_reply(
        tokenizer,
        model,
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def _run_generate_locked(
    messages: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    with _gen_lock:
        return _run_generate(messages, max_new_tokens, temperature, top_p)


def _sse_stream_locked(
    messages: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    cancel_event: threading.Event,
):
    assert _engine is not None
    tokenizer = _engine["tokenizer"]
    model = _engine["model"]
    with _gen_lock:
        for delta in generate_chat_reply_stream(
            tokenizer,
            model,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            cancel_event=cancel_event,
        ):
            line = json.dumps({"delta": delta}, ensure_ascii=False)
            yield f"data: {line}\n\n".encode("utf-8")
        if not cancel_event.is_set():
            yield b"data: [DONE]\n\n"


def _prune_completed_jobs_if_needed() -> None:
    with _jobs_lock:
        n = len(_jobs)
        if n <= _MAX_JOBS_BUFFER:
            return
        to_remove = n - _MAX_JOBS_BUFFER + 8
        finished: list[str] = []
        for k, v in _jobs.items():
            if v.get("status") in ("done", "error", "cancelled"):
                finished.append(k)
            if len(finished) >= to_remove:
                break
        for k in finished:
            _jobs.pop(k, None)


def _job_worker(
    job_id: str,
    messages: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    assert _engine is not None
    with _jobs_lock:
        st = _jobs.get(job_id)
    if not st:
        return
    cancel_event: threading.Event = st["cancel_event"]
    tokenizer = _engine["tokenizer"]
    model = _engine["model"]
    try:
        with _gen_lock:
            for delta in generate_chat_reply_stream(
                tokenizer,
                model,
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                cancel_event=cancel_event,
            ):
                with _jobs_lock:
                    if job_id not in _jobs:
                        return
                    _jobs[job_id]["text"] += delta
        with _jobs_lock:
            if job_id not in _jobs:
                return
            j = _jobs[job_id]
            if j["cancel_event"].is_set():
                j["status"] = "cancelled"
            else:
                j["status"] = "done"
    except Exception as err:  # noqa: BLE001
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = str(err)


async def _watch_client_disconnect(request: Request, cancel_event: threading.Event) -> None:
    try:
        while True:
            if await request.is_disconnected():
                cancel_event.set()
                return
            await asyncio.sleep(0.2)
    except asyncio.CancelledError:
        return


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        return HTMLResponse(
            "<p>Falta server/static/index.html</p>",
            status_code=500,
        )
    return FileResponse(index_path, media_type="text/html; charset=utf-8")


def _file_response_or_404(path: Path, media_type: str) -> FileResponse:
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{path.name} em falta.")
    return FileResponse(path, media_type=media_type)


@app.get("/app.css")
async def serve_app_css():
    """Raiz: o index em / usa href relativo app.css → /app.css."""
    return _file_response_or_404(STATIC_DIR / "app.css", "text/css; charset=utf-8")


@app.get("/app.js")
async def serve_app_js():
    return _file_response_or_404(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")


@app.get("/static/app.css")
async def serve_app_css_legacy():
    return _file_response_or_404(STATIC_DIR / "app.css", "text/css; charset=utf-8")


@app.get("/static/app.js")
async def serve_app_js_legacy():
    return _file_response_or_404(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")


@app.get("/api/status")
async def status():
    if _engine is None:
        return {"loaded": False}
    return {
        "loaded": True,
        "mode": _engine["mode"],
        "model_name": _engine["model_name"],
    }


@app.post("/api/chat", response_model=ChatOut)
async def chat(body: ChatIn):
    if _engine is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado.")
    msgs = [{"role": m.role, "content": m.content} for m in body.messages]
    for m in msgs:
        if m["role"] not in ("user", "assistant", "system"):
            raise HTTPException(status_code=400, detail="role inválido.")

    loop = asyncio.get_event_loop()
    try:
        reply = await loop.run_in_executor(
            None,
            lambda m=msgs,
            mt=body.max_new_tokens,
            t=body.temperature,
            tp=body.top_p: _run_generate_locked(m, mt, t, tp),
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err

    return ChatOut(reply=reply)


@app.post("/api/chat/stream")
async def chat_stream(request: Request, body: ChatIn):
    if _engine is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado.")
    msgs = [{"role": m.role, "content": m.content} for m in body.messages]
    for m in msgs:
        if m["role"] not in ("user", "assistant", "system"):
            raise HTTPException(status_code=400, detail="role inválido.")

    cancel_event = threading.Event()
    disconnect_task = asyncio.create_task(_watch_client_disconnect(request, cancel_event))

    def sse_iter():
        try:
            yield from _sse_stream_locked(
                msgs,
                body.max_new_tokens,
                body.temperature,
                body.top_p,
                cancel_event,
            )
        finally:
            disconnect_task.cancel()

    return StreamingResponse(
        sse_iter(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat/jobs", response_model=JobCreateOut)
async def create_chat_job(body: ChatIn):
    """Inicia geração em background; o cliente consulta com GET /api/chat/jobs/{id} (polling)."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado.")
    msgs = [{"role": m.role, "content": m.content} for m in body.messages]
    for m in msgs:
        if m["role"] not in ("user", "assistant", "system"):
            raise HTTPException(status_code=400, detail="role inválido.")
    _prune_completed_jobs_if_needed()
    job_id = secrets.token_urlsafe(16)
    cancel_event = threading.Event()
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "text": "",
            "error": None,
            "cancel_event": cancel_event,
        }
    thread = threading.Thread(
        target=_job_worker,
        args=(job_id, msgs, body.max_new_tokens, body.temperature, body.top_p),
        daemon=True,
        name=f"job-{job_id[:8]}",
    )
    thread.start()
    return JobCreateOut(job_id=job_id)


@app.get("/api/chat/jobs/{job_id}", response_model=JobStateOut)
async def get_chat_job(job_id: str):
    with _jobs_lock:
        j = _jobs.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job inexistente ou expirado.")
    return JobStateOut(
        status=j["status"],
        text=j.get("text", ""),
        error=j.get("error"),
    )


@app.post("/api/chat/jobs/{job_id}/cancel")
async def cancel_chat_job(job_id: str):
    with _jobs_lock:
        j = _jobs.get(job_id)
        if not j:
            raise HTTPException(status_code=404, detail="job inexistente.")
        j["cancel_event"].set()
    return {"ok": True}


def main() -> None:
    global _startup_args
    import uvicorn

    _startup_args = _parse_args()
    uvicorn.run(
        app,
        host=_startup_args.host,
        port=_startup_args.port,
    )


if __name__ == "__main__":
    main()
