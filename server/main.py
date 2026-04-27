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

from server.api.v1 import openai_router
from server.api.web import web_router
from server.db.auth_db import init_db, llama_upstream_base_url_from_db
from server.db.pg_db import get_pg_dsn_dict
from server.inference.runtime import get_runtime

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


def _llama_cpp_upstream_url_from_env() -> str | None:
    v = (os.environ.get("ORACULO_LLAMA_CPP_BASE_URL") or "").strip().rstrip("/")
    return v or None


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
        "  - Só interface, sem base de dados nem modelo:  ./serve.sh -- --ui-only\n"
        f"  Detalhe: {err}\n"
        "--------------------------------\n",
        file=sys.stderr,
        flush=True,
    )


def _missing_llama_server_msg() -> str:
    return (
        "Inferência: o Oráculo não carrega GGUF nem PyTorch — só delega ao llama-server.\n"
        "  Defina ORACULO_LLAMA_CPP_BASE_URL no .env (ex.: http://127.0.0.1:8080), ou\n"
        "  no admin (PostgreSQL) ative «Usar llama-server» e preencha o URL base.\n"
        "  Modo só UI:  ./serve.sh -- --ui-only"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    rt = get_runtime()
    try:
        init_db()
    except Exception as err:
        _print_pg_ligação_falhou(err)
        raise
    print("Base de utilizadores e sessões de chat: pronta.", flush=True)
    args = _startup_args if _startup_args is not None else parse_args()
    if args.ui_only:
        rt.set_ui_only()
        print("Modo --ui-only: interface e API de auth; modelo não carregado.", flush=True)
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

    db_upstream = llama_upstream_base_url_from_db()
    upstream = db_upstream or _llama_cpp_upstream_url_from_env()
    if not upstream:
        print(_missing_llama_server_msg(), file=sys.stderr, flush=True)
        raise RuntimeError(
            "Sem URL do llama-server: ORACULO_LLAMA_CPP_BASE_URL ou configuração no admin."
        )

    rt.ui_only = False
    api_key = (os.environ.get("ORACULO_LLAMA_CPP_API_KEY") or "").strip() or None
    model_ov = (os.environ.get("ORACULO_LLAMA_CPP_MODEL") or "").strip() or None
    try:
        src = "base de dados (admin)" if db_upstream else ".env ORACULO_LLAMA_CPP_BASE_URL"
        print(
            f"Modo llama-server ({src}): delegação de inferência para {upstream!r}…",
            flush=True,
        )
        rt.load_llama_server(upstream, api_key=api_key, model=model_ov)
        print(f"  Modelo remoto (id na API): {rt.model_id!r}", flush=True)
    except Exception as err:
        print(f"Erro ao ligar ao llama-server: {err}", file=sys.stderr, flush=True)
        raise
    if args.host in ("0.0.0.0", "::", "[::]"):
        print(
            f"Pronto (llama_server). À escuta em {args.host}:{args.port} — nesta máquina: "
            f"http://127.0.0.1:{args.port}/ — inferência: {upstream}/v1",
            flush=True,
        )
    else:
        print(
            f"Pronto (llama_server). Servidor em http://{args.host}:{args.port}/ — inferência: {upstream}/v1",
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
