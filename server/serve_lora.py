#!/usr/bin/env python3
"""
Servidor local (FastAPI): carrega base + LoRA **uma vez** e serve chat por HTTP.

Evita repetir o custo de carregar vários GB em cada execução de trein/inferir.py.

  cd <raiz do projeto> && python3 server/serve_lora.py
  # ou: ./server/serve.sh

Só interface (auth + estáticos, **sem** carregar o modelo): --ui-only ou ORACULO_UI_ONLY=1

Por defeito escuta em 0.0.0.0 (todas as interfaces): aceda com http://IP-DO-SERVIDOR:8765/
Nesta máquina: http://127.0.0.1:8765/ — só local: --host 127.0.0.1

API compatível com OpenAI: POST /v1/chat/completions e GET /v1/models (ver server/README.md).
A lógica de inferência está em model_service/ na raiz do repositório.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import re
import secrets
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

_SERVER_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVER_DIR.parent
_TREIN_DIR = _PROJECT_ROOT / "trein"
for _p in (_PROJECT_ROOT, _TREIN_DIR, _SERVER_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_dotenv_early() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_PROJECT_ROOT / ".env")
    load_dotenv(_SERVER_DIR / ".env", override=True)


_load_dotenv_early()

from data_config import (
    DEFAULT_ADAPTER_DIR,
    DEFAULT_MERGED_MODEL_DIR,
    DEFAULT_MODEL_NAME,
    apply_loading_progress_env,
)

apply_loading_progress_env()

if not os.environ.get("PYTORCH_ALLOC_CONF"):
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from model_service.openai_routes import router as openai_router
from model_service.runtime import get_runtime

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.sessions import SessionMiddleware

from auth_db import (
    create_user,
    get_global_system_prompt,
    get_user_model_settings,
    get_user_names,
    init_db,
    is_user_admin,
    list_all_users,
    set_global_system_prompt,
    set_user_display_name,
    set_user_email,
    set_user_model_settings,
    set_user_system_prompt_only,
    verify_user,
)
from chat_sessions_db import (
    DEFAULT_SESSION_TITLE,
    append_turn,
    create_session,
    delete_session,
    get_session_messages,
    list_sessions,
    plain_for_storage,
    set_session_title_from_model,
    set_session_title_user,
    should_generate_title,
)

STATIC_DIR = _SERVER_DIR / "static"

_user_presence_lock = threading.Lock()
_user_presence: dict[int, float] = {}
PRESENCE_ONLINE_S = 300.0


_startup_args: argparse.Namespace | None = None


def _load_session_secret() -> str:
    p = _SERVER_DIR / "data" / ".session_secret"
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

# Parâmetros de inferência para contas normais (não alteráveis via UI; o admin ajusta as suas).
_DEFAULT_MAX_NEW = 2048
_DEFAULT_TEMP = 0.7
_DEFAULT_TOP_P = 0.9

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
        help="Pasta do adapter (train_lora.py); padrão: trein/outputs/lora_adapter na raiz do repo. "
        "Ignorado se --merged_model_dir válido. ORACULO_ADAPTER_DIR no .env sobrepõe.",
    )
    p.add_argument(
        "--merged_model_dir",
        type=Path,
        default=None,
        help=f"Se existir com config.json, carrega só o modelo fundido (merge_lora.py). "
        f"Padrão: tentar {DEFAULT_MERGED_MODEL_DIR} se existir.",
    )
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument(
        "--ui-only",
        action="store_true",
        help="Não carrega o modelo: só autenticação e ficheiros estáticos. (Ou ORACULO_UI_ONLY=1.)",
    )
    args = p.parse_args()
    v = os.environ.get("ORACULO_UI_ONLY", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        args.ui_only = True
    ad_env = os.environ.get("ORACULO_ADAPTER_DIR", "").strip()
    if ad_env:
        args.adapter_dir = Path(ad_env).expanduser()
    args.adapter_dir = args.adapter_dir.resolve()
    if args.merged_model_dir is not None:
        args.merged_model_dir = Path(args.merged_model_dir).expanduser().resolve()
    return args


def _openai_key_configured() -> bool:
    return bool((os.environ.get("ORACULO_OPENAI_API_KEY") or "").strip())


def _model_unavailable_detail() -> str:
    rt = get_runtime()
    if rt.ui_only:
        return "Modo --ui-only (sem modelo). Suba o servidor sem --ui-only para o chat."
    return "Modelo ainda não carregado."


def _resolve_merged_path(cli_path: Path | None) -> Path | None:
    if cli_path is not None:
        return cli_path
    env_m = (os.environ.get("ORACULO_MERGED_MODEL_DIR") or "").strip()
    if env_m:
        p = Path(env_m).expanduser().resolve()
        if p.is_dir() and (p / "config.json").is_file():
            return p
    if DEFAULT_MERGED_MODEL_DIR.is_dir() and (DEFAULT_MERGED_MODEL_DIR / "config.json").is_file():
        return DEFAULT_MERGED_MODEL_DIR
    return None


def _print_pg_ligação_falhou(err: BaseException) -> None:
    try:
        import psycopg2
        from pg_db import get_pg_dsn_dict

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    rt = get_runtime()
    try:
        init_db()
    except Exception as err:
        _print_pg_ligação_falhou(err)
        raise
    print("Base de utilizadores e sessões de chat: pronta.", flush=True)
    args = _startup_args if _startup_args is not None else _parse_args()
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

    rt.ui_only = False
    merged = _resolve_merged_path(args.merged_model_dir)
    print("A carregar modelo (só uma vez; pode demorar)…", flush=True)
    try:
        rt.load(
            args.model_name,
            args.adapter_dir,
            merged,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as err:
        print(f"Erro ao carregar: {err}", file=sys.stderr, flush=True)
        raise
    mode = rt.mode
    if args.host in ("0.0.0.0", "::", "[::]"):
        print(
            f"Pronto ({mode}). À escuta em {args.host}:{args.port} — nesta máquina: "
            f"http://127.0.0.1:{args.port}/ — noutro PC/rede: http://<IP>:{args.port}/",
            flush=True,
        )
    else:
        print(f"Pronto ({mode}). Servidor em http://{args.host}:{args.port}/", flush=True)
    if _openai_key_configured():
        print(
            "API OpenAI-compat: POST /v1/chat/completions (Authorization: Bearer <ORACULO_OPENAI_API_KEY>).",
            flush=True,
        )
    else:
        print(
            "API OpenAI-compat: POST /v1/chat/completions (sem ORACULO_OPENAI_API_KEY — só use em rede confiável).",
            flush=True,
        )
    yield
    rt.clear()


app = FastAPI(title="Oráculo LoRA local", lifespan=lifespan)

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


class Message(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    max_new_tokens: int = Field(2048, ge=16, le=4096)
    temperature: float = Field(0.7, ge=0.01, le=2.0)
    top_p: float = Field(0.9, ge=0.05, le=1.0)


class ChatJobIn(BaseModel):
    """Pedido de job de chat: igual a ChatIn, com sessão persistida opcional."""

    messages: list[Message] = Field(..., min_length=1)
    max_new_tokens: int = Field(2048, ge=16, le=4096)
    temperature: float = Field(0.7, ge=0.01, le=2.0)
    top_p: float = Field(0.9, ge=0.05, le=1.0)
    session_id: int | None = None


class ChatOut(BaseModel):
    reply: str


class JobCreateOut(BaseModel):
    job_id: str


class JobStateOut(BaseModel):
    status: str
    text: str
    error: str | None = None
    output_tokens: int | None = None
    gen_seconds: float | None = None
    tokens_per_sec: float | None = None


class SessionTitleUpdate(BaseModel):
    title: str = Field(..., max_length=200)


class UserProfileUpdate(BaseModel):
    display_name: str = Field("", max_length=64)
    email: str = Field("", max_length=254)


class UserModelSettingsIn(BaseModel):
    system_prompt: str | None = None
    global_system_prompt: str | None = None
    max_new_tokens: int | None = Field(None, ge=16, le=4096)
    temperature: float | None = Field(None, ge=0.01, le=2.0)
    top_p: float | None = Field(None, ge=0.05, le=1.0)


class UserModelSettingsOut(BaseModel):
    is_admin: bool
    system_prompt: str
    global_system_prompt: str
    max_new_tokens: int
    temperature: float
    top_p: float


class AuthIn(BaseModel):
    username: str = Field(..., max_length=32)
    password: str = Field(..., max_length=128)

    @field_validator("username")
    @classmethod
    def _username_len(cls, v: str) -> str:
        t = (v or "").strip()
        if len(t) < 3 or len(t) > 32:
            raise ValueError("Utilizador: entre 3 e 32 caracteres.")
        return t

    @field_validator("password")
    @classmethod
    def _password_len(cls, v: str) -> str:
        if (v is None) or (len(v) < 8):
            raise ValueError("Palavra-passe: pelo menos 8 caracteres.")
        if len(v) > 128:
            raise ValueError("Palavra-passe: no máximo 128 caracteres.")
        return v


def _touch_user_presence(user_id: int) -> None:
    with _user_presence_lock:
        _user_presence[int(user_id)] = time.time()


def _require_user_id(request: Request) -> int:
    uid = request.session.get("user_id")
    if uid is None:
        raise HTTPException(status_code=401, detail="Não autenticado.")
    u = int(uid)
    _touch_user_presence(u)
    return u


def _require_admin(request: Request) -> int:
    uid = request.session.get("user_id")
    if uid is None:
        raise HTTPException(status_code=401, detail="Não autenticado.")
    u = int(uid)
    _touch_user_presence(u)
    if not is_user_admin(u):
        raise HTTPException(status_code=403, detail="Apenas administrador.")
    return u


UserIdDep = Annotated[int, Depends(_require_user_id)]
AdminIdDep = Annotated[int, Depends(_require_admin)]


def _user_settings_out(uid: int) -> UserModelSettingsOut:
    s = get_user_model_settings(int(uid))
    admin = is_user_admin(int(uid))
    g = get_global_system_prompt() if admin else ""
    if admin:
        return UserModelSettingsOut(
            is_admin=True,
            system_prompt=s["system_prompt"],
            global_system_prompt=g,
            max_new_tokens=s["max_new_tokens"],
            temperature=s["temperature"],
            top_p=s["top_p"],
        )
    return UserModelSettingsOut(
        is_admin=False,
        system_prompt=s["system_prompt"],
        global_system_prompt="",
        max_new_tokens=_DEFAULT_MAX_NEW,
        temperature=_DEFAULT_TEMP,
        top_p=_DEFAULT_TOP_P,
    )


def _merge_system_blocks(global_text: str, user_text: str) -> str | None:
    g = (global_text or "").strip()
    u = (user_text or "").strip()
    if g and u:
        return f"{g}\n\n{u}"
    if g:
        return g
    if u:
        return u
    return None


def _chat_messages_for_user(user_id: int, base: list[dict]) -> list[dict]:
    prefs = get_user_model_settings(int(user_id))
    gsp = get_global_system_prompt()
    no_client_system = [m for m in base if m.get("role") != "system"]
    usp = (prefs.get("system_prompt") or "").strip()
    merged = _merge_system_blocks(gsp, usp)
    if merged:
        return [{"role": "system", "content": merged}, *no_client_system]
    return no_client_system


def _infer_params_for_user(user_id: int) -> tuple[int, float, float]:
    if is_user_admin(int(user_id)):
        p = get_user_model_settings(int(user_id))
        return int(p["max_new_tokens"]), float(p["temperature"]), float(p["top_p"])
    return _DEFAULT_MAX_NEW, _DEFAULT_TEMP, _DEFAULT_TOP_P


@app.post("/api/auth/register")
async def auth_register(request: Request, body: AuthIn):
    if not re.match(r"^[\w.-]{3,32}$", body.username, re.IGNORECASE):
        raise HTTPException(
            status_code=400,
            detail="Utilizador: 3 a 32 caracteres (letras, números, _ . -).",
        )
    try:
        uid, uname = create_user(body.username, body.password)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    request.session["user_id"] = uid
    request.session["username"] = uname
    return {"ok": True, "username": uname}


@app.post("/api/auth/login")
async def auth_login(request: Request, body: AuthIn):
    out = verify_user(body.username, body.password)
    if not out:
        raise HTTPException(
            status_code=401, detail="Utilizador ou palavra-passe inválidos."
        )
    uid, uname = out
    request.session["user_id"] = uid
    request.session["username"] = uname
    return {"ok": True, "username": uname}


@app.post("/api/auth/logout")
async def auth_logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.get("/api/auth/me")
async def auth_me(request: Request):
    uid = request.session.get("user_id")
    if not uid:
        return {"authenticated": False}
    names = get_user_names(int(uid))
    uname = request.session.get("username", "")
    if not names:
        return {
            "authenticated": True,
            "user_id": int(uid),
            "username": uname,
            "display_name": None,
            "name": uname,
            "email": "",
            "is_admin": is_user_admin(int(uid)),
        }
    return {
        "authenticated": True,
        "user_id": int(uid),
        "username": names["username"],
        "display_name": names.get("display_name") or None,
        "name": names["name"],
        "email": names.get("email") or "",
        "is_admin": is_user_admin(int(uid)),
    }


@app.get("/api/user/profile")
async def user_get_profile(_uid: UserIdDep):
    n = get_user_names(int(_uid))
    if not n:
        raise HTTPException(status_code=404, detail="Utilizador não encontrado.")
    return {
        "username": n["username"],
        "display_name": n.get("display_name") or "",
        "name": n["name"],
        "email": n.get("email") or "",
    }


@app.patch("/api/user/profile")
async def user_patch_profile(_uid: UserIdDep, body: UserProfileUpdate):
    try:
        t = set_user_display_name(int(_uid), body.display_name)
        em = set_user_email(int(_uid), body.email)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    n = get_user_names(int(_uid)) or {"username": "", "name": t, "email": em}
    return {
        "username": n["username"],
        "display_name": t,
        "name": n["name"],
        "email": em,
    }


@app.get("/api/user/settings", response_model=UserModelSettingsOut)
async def user_get_settings(_uid: UserIdDep):
    return _user_settings_out(int(_uid))


@app.patch("/api/user/settings", response_model=UserModelSettingsOut)
async def user_patch_settings(_uid: UserIdDep, body: UserModelSettingsIn):
    uid = int(_uid)
    admin = is_user_admin(uid)
    if admin:
        if body.global_system_prompt is not None:
            set_global_system_prompt(body.global_system_prompt)
        set_user_model_settings(
            uid,
            system_prompt=body.system_prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
            top_p=body.top_p,
        )
    else:
        if body.system_prompt is not None:
            set_user_system_prompt_only(uid, body.system_prompt)
    return _user_settings_out(uid)


def _clean_title_text(raw: str) -> str:
    s = (raw or "").strip()
    s = s.split("\n", 1)[0].strip()
    s = s.strip("«»\"'“”`•-–—:")
    s = re.sub(r"\s+", " ", s)
    return s[:200] if s else ""


def _fallback_title_from_user_line(user_line: str) -> str:
    t = re.sub(r"\s+", " ", (user_line or "").strip())[:64]
    return t or "Conversa"


def _title_after_turn(user_id: int, session_id: int, last_user: str, assistant: str) -> None:
    if not should_generate_title(user_id, session_id):
        return
    rt = get_runtime()
    if rt.ui_only or not rt.is_loaded:
        set_session_title_from_model(
            user_id, session_id, _fallback_title_from_user_line(last_user)
        )
        return

    tmsgs: list[dict] = [
        {
            "role": "user",
            "content": (
                "Gera um título muito curto (máximo 6 palavras) em português para a conversa abaixo. "
                "Responde só com o título, uma linha, sem aspas e sem ponto no fim.\n\n"
                f"Utilizador: {last_user[:2000]}\n\n"
                f"Assistente: {assistant[:2000]}"
            ),
        }
    ]
    try:
        raw = rt.generate(
            tmsgs,
            max_new_tokens=64,
            temperature=0.4,
            top_p=0.9,
            user_id=int(user_id),
        )
    except Exception:
        raw = ""
    title = _clean_title_text(raw) or _fallback_title_from_user_line(last_user)
    if title and should_generate_title(user_id, session_id):
        set_session_title_from_model(user_id, session_id, title)


def _run_generate_locked(
    messages: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    user_id: int | None = None,
) -> str:
    rt = get_runtime()
    return rt.generate(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        user_id=user_id,
    )


def _sse_stream_locked(
    user_id: int,
    messages: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    cancel_event: threading.Event,
):
    rt = get_runtime()
    for delta in rt.stream(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        cancel_event=cancel_event,
        user_id=int(user_id),
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
    session_id: int | None = None,
    user_id: int | None = None,
) -> None:
    rt = get_runtime()
    assert rt.is_loaded
    with _jobs_lock:
        st = _jobs.get(job_id)
    if not st:
        return
    cancel_event: threading.Event = st["cancel_event"]
    tokenizer = rt.tokenizer
    try:
        t0: float
        t1: float
        t0 = time.perf_counter()
        for delta in rt.stream(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            cancel_event=cancel_event,
            user_id=(int(user_id) if user_id is not None else None),
        ):
            with _jobs_lock:
                if job_id not in _jobs:
                    return
                _jobs[job_id]["text"] += delta
        t1 = time.perf_counter()
        with _jobs_lock:
            if job_id not in _jobs:
                return
            j = _jobs[job_id]
            asst_full = j.get("text") or ""
            gen_sec = max(float(t1 - t0), 1e-9)
            if asst_full.strip():
                toks = len(
                    tokenizer.encode(str(asst_full), add_special_tokens=False)  # type: ignore[union-attr]
                )
            else:
                toks = 0
            j["output_tokens"] = toks
            j["gen_seconds"] = round(float(t1 - t0), 2)
            j["tokens_per_sec"] = round(toks / gen_sec, 2) if toks else 0.0
            if j["cancel_event"].is_set():
                j["status"] = "cancelled"
            else:
                j["status"] = "done"
            is_done = j["status"] == "done" and not j["cancel_event"].is_set()
            asst_text = (j.get("text") or "") if is_done else ""
            persist_toks = toks
            persist_sec = float(j["gen_seconds"])
            persist_tps = float(j["tokens_per_sec"])
        if is_done and session_id and user_id and asst_text.strip():
            last_u = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    last_u = str(m.get("content", ""))
                    break
            if last_u and append_turn(
                int(user_id),
                int(session_id),
                last_u,
                asst_text,
                output_tokens=persist_toks,
                gen_seconds=persist_sec,
                tokens_per_sec=persist_tps,
            ):
                u_pl = plain_for_storage(last_u)
                a_pl = plain_for_storage(asst_text)
                threading.Thread(
                    target=_title_after_turn,
                    args=(int(user_id), int(session_id), u_pl, a_pl),
                    daemon=True,
                ).start()
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


@app.get("/")
async def index(request: Request):
    if not request.session.get("user_id"):
        return RedirectResponse(url="/login", status_code=302)
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        return HTMLResponse(
            "<p>Falta server/static/index.html</p>",
            status_code=500,
        )
    return FileResponse(index_path, media_type="text/html; charset=utf-8")


@app.get("/login", response_class=HTMLResponse)
async def page_login(request: Request):
    if request.session.get("user_id"):
        return RedirectResponse(url="/", status_code=302)
    p = STATIC_DIR / "login.html"
    if not p.is_file():
        return HTMLResponse(
            "<p>Falta server/static/login.html</p>", status_code=500
        )
    return FileResponse(p, media_type="text/html; charset=utf-8")


@app.get("/registar", response_class=HTMLResponse)
async def page_registar(request: Request):
    if request.session.get("user_id"):
        return RedirectResponse(url="/", status_code=302)
    p = STATIC_DIR / "registar.html"
    if not p.is_file():
        return HTMLResponse(
            "<p>Falta server/static/registar.html</p>", status_code=500
        )
    return FileResponse(p, media_type="text/html; charset=utf-8")


@app.get("/admin")
async def page_admin(request: Request):
    uid = request.session.get("user_id")
    if not uid:
        return RedirectResponse(url="/login", status_code=302)
    if not is_user_admin(int(uid)):
        return RedirectResponse(url="/", status_code=302)
    p = STATIC_DIR / "admin.html"
    if not p.is_file():
        return HTMLResponse(
            "<p>Falta server/static/admin.html</p>",
            status_code=500,
        )
    return FileResponse(p, media_type="text/html; charset=utf-8")


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


@app.get("/admin.js")
async def serve_admin_js():
    return _file_response_or_404(STATIC_DIR / "admin.js", "application/javascript; charset=utf-8")


@app.get("/static/app.css")
async def serve_app_css_legacy():
    return _file_response_or_404(STATIC_DIR / "app.css", "text/css; charset=utf-8")


@app.get("/static/app.js")
async def serve_app_js_legacy():
    return _file_response_or_404(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")


@app.get("/api/status")
async def status(_uid: UserIdDep):
    return get_runtime().status_public()


@app.get("/api/admin/users")
async def api_admin_list_users(_admin: AdminIdDep) -> dict[str, list]:
    now = time.time()
    with _user_presence_lock:
        pres = dict(_user_presence)
    gen_uid = get_runtime().active_generation_user_id
    users = list_all_users()
    out: list[dict[str, Any]] = []
    for u in users:
        uidi = int(u["id"])
        last = pres.get(uidi, 0.0)
        out.append(
            {
                **u,
                "online": (now - last) < PRESENCE_ONLINE_S and last > 0,
                "using_server": gen_uid is not None and int(gen_uid) == uidi,
            }
        )
    return {"users": out}


@app.get("/api/admin/users/{target_id}/sessions")
async def api_admin_user_sessions(
    _admin: AdminIdDep, target_id: int
) -> dict[str, list]:
    if target_id < 1:
        raise HTTPException(status_code=400, detail="id inválido.")
    return {"sessions": list_sessions(int(target_id))}


@app.get("/api/admin/users/{target_id}/sessions/{session_id}")
async def api_admin_user_session(
    _admin: AdminIdDep, target_id: int, session_id: int
) -> dict:
    if target_id < 1 or session_id < 1:
        raise HTTPException(status_code=400, detail="id inválido.")
    out = get_session_messages(int(target_id), int(session_id))
    if not out:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")
    messages, title, session_meta = out
    return {
        "id": session_id,
        "user_id": target_id,
        "title": title,
        "messages": messages,
        **session_meta,
    }


@app.get("/api/chat/generation-status")
async def api_chat_generation_status(_uid: UserIdDep) -> dict[str, bool]:
    """
    Se `active` e não `yours`, outro utilizador detém a inferência (fila de um pedido de cada vez).
    """
    rt = get_runtime()
    if rt.ui_only or not rt.is_loaded:
        return {"active": False, "yours": False}
    uid = int(_uid)
    holder = rt.active_generation_user_id
    if holder is None:
        return {"active": False, "yours": False}
    return {"active": True, "yours": holder == uid}


@app.post("/api/chat", response_model=ChatOut)
async def chat(_uid: UserIdDep, body: ChatIn):
    if not get_runtime().is_loaded:
        raise HTTPException(status_code=503, detail=_model_unavailable_detail())
    base = [{"role": m.role, "content": m.content} for m in body.messages]
    for m in base:
        if m["role"] not in ("user", "assistant", "system"):
            raise HTTPException(status_code=400, detail="role inválido.")
    uid = int(_uid)
    msgs = _chat_messages_for_user(uid, base)
    mnt, temp, top_p = _infer_params_for_user(uid)

    loop = asyncio.get_event_loop()
    try:
        reply = await loop.run_in_executor(
            None,
            lambda m=msgs, mt=mnt, t=temp, tp=top_p, u=uid: _run_generate_locked(
                m, mt, t, tp, u
            ),
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err)) from err

    return ChatOut(reply=reply)


@app.post("/api/chat/stream")
async def chat_stream(request: Request, body: ChatIn, _uid: UserIdDep):
    if not get_runtime().is_loaded:
        raise HTTPException(status_code=503, detail=_model_unavailable_detail())
    base = [{"role": m.role, "content": m.content} for m in body.messages]
    for m in base:
        if m["role"] not in ("user", "assistant", "system"):
            raise HTTPException(status_code=400, detail="role inválido.")
    uid = int(_uid)
    msgs = _chat_messages_for_user(uid, base)
    mnt, temp, top_p = _infer_params_for_user(uid)

    cancel_event = threading.Event()
    disconnect_task = asyncio.create_task(_watch_client_disconnect(request, cancel_event))

    def sse_iter():
        try:
            yield from _sse_stream_locked(
                uid,
                msgs,
                mnt,
                temp,
                top_p,
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


@app.get("/api/sessions")
async def api_list_sessions(_uid: UserIdDep):
    return {"sessions": list_sessions(int(_uid))}


@app.post("/api/sessions")
async def api_new_session(_uid: UserIdDep):
    sid = create_session(int(_uid))
    return {"id": sid, "title": DEFAULT_SESSION_TITLE}


@app.get("/api/sessions/{session_id}")
async def api_get_session(_uid: UserIdDep, session_id: int):
    out = get_session_messages(int(_uid), session_id)
    if out is None:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")
    messages, title, session_meta = out
    return {"id": session_id, "title": title, "messages": messages, **session_meta}


@app.patch("/api/sessions/{session_id}")
async def api_rename_session(
    _uid: UserIdDep, session_id: int, body: SessionTitleUpdate
):
    t = (body.title or "").strip()
    if not t:
        raise HTTPException(status_code=400, detail="Título vazio.")
    if not set_session_title_user(int(_uid), session_id, t):
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")
    return {"id": session_id, "title": t[:200]}


@app.delete("/api/sessions/{session_id}")
async def api_delete_session(_uid: UserIdDep, session_id: int):
    if not delete_session(int(_uid), session_id):
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")
    return {"ok": True}


@app.post("/api/chat/jobs", response_model=JobCreateOut)
async def create_chat_job(_uid: UserIdDep, body: ChatJobIn):
    """Inicia geração em background; o cliente consulta com GET /api/chat/jobs/{id} (polling)."""
    if not get_runtime().is_loaded:
        raise HTTPException(status_code=503, detail=_model_unavailable_detail())
    if body.session_id is not None:
        if get_session_messages(int(_uid), body.session_id) is None:
            raise HTTPException(status_code=404, detail="Sessão não encontrada.")
    base = [{"role": m.role, "content": m.content} for m in body.messages]
    for m in base:
        if m["role"] not in ("user", "assistant", "system"):
            raise HTTPException(status_code=400, detail="role inválido.")
    uid = int(_uid)
    msgs = _chat_messages_for_user(uid, base)
    mnt, temp, tp = _infer_params_for_user(uid)
    _prune_completed_jobs_if_needed()
    job_id = secrets.token_urlsafe(16)
    cancel_event = threading.Event()
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "text": "",
            "error": None,
            "cancel_event": cancel_event,
            "user_id": uid,
        }
    session_id: int | None = body.session_id
    user_id: int = int(_uid)
    thread = threading.Thread(
        target=_job_worker,
        args=(
            job_id,
            msgs,
            mnt,
            temp,
            tp,
            session_id,
            user_id,
        ),
        daemon=True,
        name=f"job-{job_id[:8]}",
    )
    thread.start()
    return JobCreateOut(job_id=job_id)


@app.get("/api/chat/jobs/{job_id}", response_model=JobStateOut)
async def get_chat_job(_uid: UserIdDep, job_id: str):
    with _jobs_lock:
        j = _jobs.get(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="job inexistente ou expirado.")
    return JobStateOut(
        status=j["status"],
        text=j.get("text", ""),
        error=j.get("error"),
        output_tokens=j.get("output_tokens"),
        gen_seconds=j.get("gen_seconds"),
        tokens_per_sec=j.get("tokens_per_sec"),
    )


@app.post("/api/chat/jobs/{job_id}/cancel")
async def cancel_chat_job(_uid: UserIdDep, job_id: str):
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
