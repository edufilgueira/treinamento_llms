#!/usr/bin/env python3
"""
Servidor local (FastAPI): carrega base + LoRA **uma vez** e serve chat por HTTP.

Evita repetir o custo de carregar vários GB em cada execução de inferir.py.

  cd <raiz do projeto> && python3 server/serve_lora.py
  # ou: ./server/serve.sh

Abre no navegador: http://127.0.0.1:8765/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

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

from fastapi import FastAPI, HTTPException
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Servidor local para o modelo LoRA treinado.")
    p.add_argument("--host", default="127.0.0.1", help="Interface (padrão: só local).")
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


class Message(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    max_new_tokens: int = Field(256, ge=16, le=4096)
    temperature: float = Field(0.7, ge=0.01, le=2.0)
    top_p: float = Field(0.9, ge=0.05, le=1.0)


class ChatOut(BaseModel):
    reply: str


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
        ):
            line = json.dumps({"delta": delta}, ensure_ascii=False)
            yield f"data: {line}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        return HTMLResponse(
            "<p>Falta server/static/index.html</p>",
            status_code=500,
        )
    return FileResponse(index_path, media_type="text/html; charset=utf-8")


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
async def chat_stream(body: ChatIn):
    if _engine is None:
        raise HTTPException(status_code=503, detail="Modelo ainda não carregado.")
    msgs = [{"role": m.role, "content": m.content} for m in body.messages]
    for m in msgs:
        if m["role"] not in ("user", "assistant", "system"):
            raise HTTPException(status_code=400, detail="role inválido.")

    return StreamingResponse(
        _sse_stream_locked(
            msgs,
            body.max_new_tokens,
            body.temperature,
            body.top_p,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
