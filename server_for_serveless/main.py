"""
Aplicação FastAPI Oráculo: arranque do modelo, middleware, rotas /api e /v1.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
import psycopg2

from server_for_serveless.api.v1 import openai_router
from server_for_serveless.api.web import web_router
from server_for_serveless.db.auth_db import get_global_runtime_prefs, init_db
from server_for_serveless.db.pg_db import get_pg_dsn_dict
from server_for_serveless.inference.bootstrap import load_inference_backend
from server_for_serveless.inference.runtime import (
    cross_user_ui_block_enabled,
    get_runtime,
    inference_single_flight_enabled,
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

_SERVER_ROOT = Path(__file__).resolve().parent

_startup_args: argparse.Namespace | None = None


def set_startup_args(args: argparse.Namespace) -> None:
    global _startup_args
    _startup_args = args


def _load_session_secret() -> str:
    import secrets

    p = _SERVER_ROOT / "data" / ".session_secret"
    env = os.environ.get("ORACULO_SESSION_SECRET", "").strip()
    if env:
        return env
    if p.is_file():
        return p.read_text().strip()
    s = secrets.token_urlsafe(32)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")
    try:
        p.chmod(0o600)
    except OSError:
        pass
    return s


SESSION_SECRET = _load_session_secret()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Servidor Oráculo (chat + PostgreSQL).")
    _bind_default = (os.environ.get("ORACULO_BIND_HOST") or "").strip() or "0.0.0.0"
    p.add_argument("--host", default=_bind_default)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--ui-only",
        action="store_true",
        help="Só auth e estáticos (ORACULO_UI_ONLY=1).",
    )
    args = p.parse_args()
    v = os.environ.get("ORACULO_UI_ONLY", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        args.ui_only = True
    return args


def _openai_key_configured() -> bool:
    return bool((os.environ.get("ORACULO_OPENAI_API_KEY") or "").strip())


def _print_pg_ligação_falhou(err: BaseException) -> None:
    try:
        if not isinstance(err, psycopg2.OperationalError):
            return
        p = get_pg_dsn_dict()
        h, port = p.get("host"), p.get("port")
        db, user = p.get("dbname", "?"), p.get("user", "?")
    except Exception:
        return
    print(
        "\n--- PostgreSQL: ligação falhou ---\n"
        f"  Tentativa: host={h!r}  port={port}  database={db!r}  user={user!r}\n"
        "  Isto costuma acontecer se a porta 5432 não estiver acessível a este PC, se o\n"
        "  servidor só aceitar ligações locais, ou se o `ORACULO_PG_HOST` no `.env` estiver errado.\n"
        "  - Na VPS: regra de firewall/Hostinger; `listen_addresses` e `pg_hba.conf` (IP permitido).\n"
        "  - Túnel SSH (exemplo):  ssh -L 5433:127.0.0.1:5432 user@servidor  "
        "e no `.env`  ORACULO_PG_HOST=127.0.0.1  ORACULO_PG_PORT=5433\n"
        "  - Só interface, sem base de dados nem modelo:  ./server_for_serveless/serve.sh -- --ui-only  "
        "ou configuração global «Só interface» (admin).\n"
        f"  Detalhe: {err}\n"
        "--------------------------------\n",
        file=sys.stderr,
        flush=True,
    )


async def lifespan(app: FastAPI):
    rt = get_runtime()
    try:
        init_db()
    except Exception as err:
        _print_pg_ligação_falhou(err)
        raise
    print("Base de utilizadores e sessões de chat: pronta.", flush=True)
    args = _startup_args if _startup_args is not None else parse_args()
    prefs = get_global_runtime_prefs()
    rt.apply_inference_queue_prefs(
        inference_single_flight=prefs["inference_single_flight"],
        ui_block_cross_user_generation=prefs["ui_block_cross_user_generation"],
    )
    env_ui = os.environ.get("ORACULO_UI_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    effective_ui_only = bool(prefs["ui_only"] or args.ui_only or env_ui)

    if effective_ui_only:
        rt.set_ui_only()
        print("Modo só interface (auth + estáticos): modelo não carregado.", flush=True)
        src = []
        if prefs["ui_only"]:
            src.append("configuração global (admin)")
        if args.ui_only:
            src.append("--ui-only")
        if env_ui:
            src.append("ORACULO_UI_ONLY")
        if src:
            print(f"  Motivo: {' e '.join(src)}.", flush=True)
        if args.host in ("0.0.0.0", "::", "[::]"):
            print(
                f"Pronto. À escuta em {args.host}:{args.port} — nesta máquina: "
                f"http://127.0.0.1:{args.port}/",
                flush=True,
            )
        else:
            print(f"Pronto. Servidor em http://{args.host}:{args.port}/", flush=True)
        yield
        rt.ui_only = False
        return

    try:
        load_inference_backend(rt)
    except Exception as err:
        print(f"Erro ao configurar inferência: {err}", file=sys.stderr, flush=True)
        raise

    if rt.backend == "runpod":
        eid = rt.runpod_endpoint_id or "?"
        print(
            "Modo runpod: inferência no Runpod Serverless; tokens/temp/top_p seguem o admin "
            "(app_global) e preferências de utilizador.",
            flush=True,
        )
        print(f"  Endpoint: {eid!r} — model id (API): {rt.model_id!r}", flush=True)
    else:
        print(
            f"Modo llama-server: delegação de inferência para {(rt.upstream_base or '')!r}…",
            flush=True,
        )
        print(f"  Modelo remoto (id na API): {rt.model_id!r}", flush=True)

    if inference_single_flight_enabled():
        print(
            "  Inferência em fila global: uma geração activa de cada vez (configuração global, admin).",
            flush=True,
        )
        if cross_user_ui_block_enabled():
            print(
                "  UI: aviso e bloqueio de envio enquanto outro utilizador gera (admin).",
                flush=True,
            )
        else:
            print(
                "  UI: sem aviso/bloqueio entre utilizadores (admin); pedidos ainda serializam no servidor.",
                flush=True,
            )
    else:
        print(
            "  Inferência concorrente: vários utilizadores em paralelo (configuração global, admin).",
            flush=True,
        )

    if args.host in ("0.0.0.0", "::", "[::]"):
        if rt.backend == "runpod":
            eid = rt.runpod_endpoint_id or "?"
            infer_note = f"inferência Runpod Serverless ({eid})"
        else:
            infer_note = f"inferência: {(rt.upstream_base or '')}/v1"
        print(
            f"Pronto ({rt.backend}). À escuta em {args.host}:{args.port} — nesta máquina: "
            f"http://127.0.0.1:{args.port}/ — {infer_note}",
            flush=True,
        )
    else:
        print(
            f"Pronto ({rt.backend}). Servidor em http://{args.host}:{args.port}/",
            flush=True,
        )
    if _openai_key_configured():
        print(
            "API OpenAI-compat: POST /v1/chat/completions (Authorization: Bearer <ORACULO_OPENAI_API_KEY>).",
            flush=True,
        )
    else:
        print(
            "API OpenAI-compat: POST /v1/chat/completions (sem ORACULO_OPENAI_API_KEY — só em rede confiável).",
            flush=True,
        )
    yield
    rt.clear()


app = FastAPI(title="Oráculo", lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
    https_only=False,
    max_age=14 * 24 * 60 * 60,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(openai_router)
app.include_router(web_router)


def main() -> None:
    import uvicorn

    args = parse_args()
    set_startup_args(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
