"""Rotas da aplicação web: auth, chat, sessões, estáticos."""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse

from server.api.deps import AdminIdDep, PRESENCE_ONLINE_S, UserIdDep, presence_snapshot
from server.db.auth_db import (
    create_user,
    get_global_system_prompt,
    get_llama_server_settings,
    get_user_model_settings,
    get_user_names,
    is_user_admin,
    list_all_users,
    set_global_system_prompt,
    set_llama_server_settings,
    set_user_display_name,
    set_user_email,
    set_user_model_settings,
    set_user_system_prompt_only,
    verify_user,
)
from server.db.chat_sessions_db import (
    DEFAULT_SESSION_TITLE,
    append_turn,
    create_session,
    delete_session,
    get_session_messages,
    list_sessions,
    plain_for_storage,
    set_assistant_message_admin_reviewed,
    set_session_title_from_model,
    set_session_title_user,
    should_generate_title,
)
from server.inference.runtime import get_runtime
from server.schemas.web import (
    AdminMessageReviewUpdate,
    AuthIn,
    ChatIn,
    ChatJobIn,
    ChatOut,
    JobCreateOut,
    JobStateOut,
    LlamaServerSettingsOut,
    SessionTitleUpdate,
    UserModelSettingsIn,
    UserModelSettingsOut,
    UserProfileUpdate,
)
from server.services.chat_prefs import (
    DEFAULT_MAX_NEW,
    DEFAULT_TEMP,
    DEFAULT_TOP_P,
    chat_messages_for_user,
    infer_params_for_user,
    model_unavailable_detail,
)

_SERVER_ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = _SERVER_ROOT / "static"

router = APIRouter()

_jobs_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}
_MAX_JOBS_BUFFER = 64


def _user_settings_out(uid: int) -> UserModelSettingsOut:
    s = get_user_model_settings(int(uid))
    admin = is_user_admin(int(uid))
    g = get_global_system_prompt() if admin else ""
    llama_out: LlamaServerSettingsOut | None = None
    if admin:
        ls = get_llama_server_settings()
        llama_out = LlamaServerSettingsOut(
            upstream_enabled=bool(ls["upstream_enabled"]),
            api_host=str(ls["api_host"]),
            api_port=int(ls["api_port"]),
            n_ctx=int(ls["n_ctx"]),
            max_new_tokens=int(ls["max_new_tokens"]),
            temperature=float(ls["temperature"]),
            top_p=float(ls["top_p"]),
            repeat_penalty=float(ls["repeat_penalty"]),
            repeat_last_n=int(ls["repeat_last_n"]),
            reasoning=str(ls["reasoning"]),
            reasoning_budget=int(ls["reasoning_budget"]),
        )
        return UserModelSettingsOut(
            is_admin=True,
            system_prompt=s["system_prompt"],
            global_system_prompt=g,
            max_new_tokens=int(ls["max_new_tokens"]),
            temperature=float(ls["temperature"]),
            top_p=float(ls["top_p"]),
            llama_server=llama_out,
        )
    rt = get_runtime()
    if rt.backend == "llama_server" and rt.is_loaded:
        ls = get_llama_server_settings()
        return UserModelSettingsOut(
            is_admin=False,
            system_prompt=s["system_prompt"],
            global_system_prompt="",
            max_new_tokens=int(ls["max_new_tokens"]),
            temperature=float(ls["temperature"]),
            top_p=float(ls["top_p"]),
            llama_server=None,
        )
    return UserModelSettingsOut(
        is_admin=False,
        system_prompt=s["system_prompt"],
        global_system_prompt="",
        max_new_tokens=DEFAULT_MAX_NEW,
        temperature=DEFAULT_TEMP,
        top_p=DEFAULT_TOP_P,
        llama_server=None,
    )


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
                toks = rt.count_output_tokens(str(asst_full))
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


def _file_response_or_404(path: Path, media_type: str) -> FileResponse:
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{path.name} em falta.")
    return FileResponse(path, media_type=media_type)


@router.post("/api/auth/register")
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


@router.post("/api/auth/login")
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


@router.post("/api/auth/logout")
async def auth_logout(request: Request):
    request.session.clear()
    return {"ok": True}


@router.get("/api/auth/me")
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


@router.get("/api/user/profile")
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


@router.patch("/api/user/profile")
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


@router.get("/api/user/settings", response_model=UserModelSettingsOut)
async def user_get_settings(_uid: UserIdDep):
    return _user_settings_out(int(_uid))


@router.patch("/api/user/settings", response_model=UserModelSettingsOut)
async def user_patch_settings(_uid: UserIdDep, body: UserModelSettingsIn):
    uid = int(_uid)
    admin = is_user_admin(uid)
    if admin:
        if body.global_system_prompt is not None:
            set_global_system_prompt(body.global_system_prompt)
        set_user_model_settings(uid, system_prompt=body.system_prompt)
        if body.llama is not None:
            patch = body.llama.model_dump(exclude_unset=True)
            if patch:
                try:
                    set_llama_server_settings(**patch)
                except ValueError as err:
                    raise HTTPException(status_code=400, detail=str(err)) from err
    else:
        if body.system_prompt is not None:
            set_user_system_prompt_only(uid, body.system_prompt)
    return _user_settings_out(uid)


@router.get("/")
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


@router.get("/login", response_class=HTMLResponse)
async def page_login(request: Request):
    if request.session.get("user_id"):
        return RedirectResponse(url="/", status_code=302)
    p = STATIC_DIR / "login.html"
    if not p.is_file():
        return HTMLResponse(
            "<p>Falta server/static/login.html</p>", status_code=500
        )
    return FileResponse(p, media_type="text/html; charset=utf-8")


@router.get("/registar", response_class=HTMLResponse)
async def page_registar(request: Request):
    if request.session.get("user_id"):
        return RedirectResponse(url="/", status_code=302)
    p = STATIC_DIR / "registar.html"
    if not p.is_file():
        return HTMLResponse(
            "<p>Falta server/static/registar.html</p>", status_code=500
        )
    return FileResponse(p, media_type="text/html; charset=utf-8")


@router.get("/admin")
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


@router.get("/app.css")
async def serve_app_css():
    return _file_response_or_404(STATIC_DIR / "app.css", "text/css; charset=utf-8")


@router.get("/app.js")
async def serve_app_js():
    return _file_response_or_404(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")


@router.get("/admin.js")
async def serve_admin_js():
    return _file_response_or_404(STATIC_DIR / "admin.js", "application/javascript; charset=utf-8")


@router.get("/static/app.css")
async def serve_app_css_legacy():
    return _file_response_or_404(STATIC_DIR / "app.css", "text/css; charset=utf-8")


@router.get("/static/app.js")
async def serve_app_js_legacy():
    return _file_response_or_404(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")


@router.get("/api/status")
async def status(_uid: UserIdDep):
    return get_runtime().status_public()


@router.get("/api/admin/users")
async def api_admin_list_users(_admin: AdminIdDep) -> dict[str, list]:
    now = time.time()
    pres = presence_snapshot()
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


@router.get("/api/admin/users/{target_id}/sessions")
async def api_admin_user_sessions(
    _admin: AdminIdDep, target_id: int
) -> dict[str, list]:
    if target_id < 1:
        raise HTTPException(status_code=400, detail="id inválido.")
    return {"sessions": list_sessions(int(target_id))}


@router.get("/api/admin/users/{target_id}/sessions/{session_id}")
async def api_admin_user_session(
    _admin: AdminIdDep, target_id: int, session_id: int
) -> dict:
    if target_id < 1 or session_id < 1:
        raise HTTPException(status_code=400, detail="id inválido.")
    out = get_session_messages(
        int(target_id), int(session_id), include_message_admin_meta=True
    )
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


@router.patch("/api/admin/users/{target_id}/sessions/{session_id}/messages/{message_id}/review")
async def api_admin_mark_message_reviewed(
    _admin: AdminIdDep,
    target_id: int,
    session_id: int,
    message_id: int,
    body: AdminMessageReviewUpdate,
) -> dict[str, bool]:
    if target_id < 1 or session_id < 1 or message_id < 1:
        raise HTTPException(status_code=400, detail="id inválido.")
    ok = set_assistant_message_admin_reviewed(
        int(target_id),
        int(session_id),
        int(message_id),
        bool(body.reviewed),
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Mensagem não encontrada.")
    return {"ok": True}


@router.get("/api/chat/generation-status")
async def api_chat_generation_status(_uid: UserIdDep) -> dict[str, bool]:
    rt = get_runtime()
    if rt.ui_only or not rt.is_loaded:
        return {"active": False, "yours": False}
    uid = int(_uid)
    holder = rt.active_generation_user_id
    if holder is None:
        return {"active": False, "yours": False}
    return {"active": True, "yours": holder == uid}


@router.post("/api/chat", response_model=ChatOut)
async def chat(_uid: UserIdDep, body: ChatIn):
    if not get_runtime().is_loaded:
        raise HTTPException(status_code=503, detail=model_unavailable_detail())
    base = [{"role": m.role, "content": m.content} for m in body.messages]
    for m in base:
        if m["role"] not in ("user", "assistant", "system"):
            raise HTTPException(status_code=400, detail="role inválido.")
    uid = int(_uid)
    msgs = chat_messages_for_user(uid, base)
    mnt, temp, top_p = infer_params_for_user(uid)

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


@router.post("/api/chat/stream")
async def chat_stream(request: Request, body: ChatIn, _uid: UserIdDep):
    if not get_runtime().is_loaded:
        raise HTTPException(status_code=503, detail=model_unavailable_detail())
    base = [{"role": m.role, "content": m.content} for m in body.messages]
    for m in base:
        if m["role"] not in ("user", "assistant", "system"):
            raise HTTPException(status_code=400, detail="role inválido.")
    uid = int(_uid)
    msgs = chat_messages_for_user(uid, base)
    mnt, temp, top_p = infer_params_for_user(uid)

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


@router.get("/api/sessions")
async def api_list_sessions(_uid: UserIdDep):
    return {"sessions": list_sessions(int(_uid))}


@router.post("/api/sessions")
async def api_new_session(_uid: UserIdDep):
    sid = create_session(int(_uid))
    return {"id": sid, "title": DEFAULT_SESSION_TITLE}


@router.get("/api/sessions/{session_id}")
async def api_get_session(_uid: UserIdDep, session_id: int):
    out = get_session_messages(int(_uid), session_id)
    if out is None:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")
    messages, title, session_meta = out
    return {"id": session_id, "title": title, "messages": messages, **session_meta}


@router.patch("/api/sessions/{session_id}")
async def api_rename_session(
    _uid: UserIdDep, session_id: int, body: SessionTitleUpdate
):
    t = (body.title or "").strip()
    if not t:
        raise HTTPException(status_code=400, detail="Título vazio.")
    if not set_session_title_user(int(_uid), session_id, t):
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")
    return {"id": session_id, "title": t[:200]}


@router.delete("/api/sessions/{session_id}")
async def api_delete_session(_uid: UserIdDep, session_id: int):
    if not delete_session(int(_uid), session_id):
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")
    return {"ok": True}


@router.post("/api/chat/jobs", response_model=JobCreateOut)
async def create_chat_job(_uid: UserIdDep, body: ChatJobIn):
    if not get_runtime().is_loaded:
        raise HTTPException(status_code=503, detail=model_unavailable_detail())
    if body.session_id is not None:
        if get_session_messages(int(_uid), body.session_id) is None:
            raise HTTPException(status_code=404, detail="Sessão não encontrada.")
    base = [{"role": m.role, "content": m.content} for m in body.messages]
    for m in base:
        if m["role"] not in ("user", "assistant", "system"):
            raise HTTPException(status_code=400, detail="role inválido.")
    uid = int(_uid)
    msgs = chat_messages_for_user(uid, base)
    mnt, temp, tp = infer_params_for_user(uid)
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


@router.get("/api/chat/jobs/{job_id}", response_model=JobStateOut)
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


@router.post("/api/chat/jobs/{job_id}/cancel")
async def cancel_chat_job(_uid: UserIdDep, job_id: str):
    with _jobs_lock:
        j = _jobs.get(job_id)
        if not j:
            raise HTTPException(status_code=404, detail="job inexistente.")
        j["cancel_event"].set()
    return {"ok": True}
