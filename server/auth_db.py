"""
Utilizadores: PostgreSQL (var ORACULO_PG_*), passwords com bcrypt + SHA-256 de pré-hash.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Any

import bcrypt
import psycopg2

from pg_db import get_connection, init_schema

_db_lock = threading.Lock()
_SERVER_DIR = Path(__file__).resolve().parent

# Texto inicial em app_global; o admin pode editar.
_DEFAULT_GLOBAL_SYSTEM_PROMPT = (
    "[SISTEMA GLOBAL — Oráculo Kiaiá, definido pelo administrador; alinha-se a todos os diálogos.] "
    "Sê claro, respeitoso e seguro. Responde na mesma língua que o utilizador."
)
_BOOTSTRAP_ADMIN_USERNAME = "admin"
# Pedido original: v03admin% (o ponto no texto era fim de frase, não parte da password).
_BOOTSTRAP_ADMIN_PASSWORD = "v03admin%"


def get_db_path() -> Path:
    """Legado: pasta de dados locais (ex.: ficheiro de sessão); não indica a DB.""" 
    return _SERVER_DIR / "data"


def _password_key(password: str) -> bytes:
    return hashlib.sha256((password or "").encode("utf-8")).digest()


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(_password_key(password), bcrypt.gensalt()).decode("ascii")


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        return bcrypt.checkpw(
            _password_key(password),
            stored_hash.encode("ascii"),
        )
    except (ValueError, TypeError, UnicodeError):
        return False


def init_db() -> None:
    (_SERVER_DIR / "data").mkdir(parents=True, exist_ok=True)
    with _db_lock:
        init_schema()
    ensure_bootstrap_admin()


def _normalize_username(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        raise ValueError("utilizador vazio")
    return s


def create_user(username: str, password: str) -> tuple[int, str]:
    u = _normalize_username(username)
    if len(u) < 3 or len(u) > 32:
        raise ValueError("utilizador: entre 3 e 32 caracteres")
    h = _hash_password(password)
    with _db_lock:
        con = get_connection()
        try:
            with con:
                with con.cursor() as cur:
                    try:
                        cur.execute(
                            """
                            INSERT INTO users (username, password_hash, is_admin)
                            VALUES (%s, %s, 0)
                            RETURNING id
                            """,
                            (u, h),
                        )
                        row = cur.fetchone()
                        if not row:
                            raise ValueError("falha ao criar utilizador")
                        return int(row[0]), u
                    except psycopg2.IntegrityError as err:
                        raise ValueError("este nome de utilizador já está em uso") from err
        finally:
            con.close()


def verify_user(username: str, password: str) -> tuple[int, str] | None:
    u = _normalize_username(username)
    with _db_lock:
        con = get_connection()
        try:
            with con.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, username, password_hash FROM users
                    WHERE LOWER(username) = LOWER(%s)
                    """,
                    (u,),
                )
                row = cur.fetchone()
        finally:
            con.close()
    if not row:
        return None
    uid, uname, ph = int(row[0]), str(row[1]), str(row[2])
    if not _verify_password(password, ph):
        return None
    return (uid, uname)


def get_user_names(user_id: int) -> dict[str, str] | None:
    """username, display_name, email e name para /api e perfil; None se o id não existir."""
    with _db_lock:
        con = get_connection()
        try:
            with con.cursor() as cur:
                cur.execute(
                    "SELECT username, display_name, email FROM users WHERE id = %s",
                    (int(user_id),),
                )
                row = cur.fetchone()
        finally:
            con.close()
    if not row:
        return None
    u = str(row[0])
    d = "" if row[1] is None else str(row[1]).strip()
    e = "" if row[2] is None else str(row[2]).strip()
    return {"username": u, "display_name": d, "name": d if d else u, "email": e}


def set_user_display_name(user_id: int, display_name: str) -> str:
    t = re.sub(r"\s+", " ", (display_name or "").strip())[:64]
    if not t:
        t = ""  # limpar → uso do username no UI
    with _db_lock:
        con = get_connection()
        try:
            with con:
                with con.cursor() as cur:
                    val = t if t else None
                    cur.execute(
                        "UPDATE users SET display_name = %s WHERE id = %s",
                        (val, int(user_id)),
                    )
                    if cur.rowcount < 1:
                        raise ValueError("utilizador inexistente")
        finally:
            con.close()
    return t


def _normalize_email(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    if len(s) > 254:
        raise ValueError("email: no máximo 254 caracteres")
    if not re.match(
        r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$",
        s,
    ):
        raise ValueError("formato de email inválido")
    return s


def set_user_email(user_id: int, email: str) -> str:
    t = _normalize_email(email)
    with _db_lock:
        con = get_connection()
        try:
            with con:
                with con.cursor() as cur:
                    cur.execute(
                        "UPDATE users SET email = %s WHERE id = %s",
                        (t or None, int(user_id)),
                    )
                    if cur.rowcount < 1:
                        raise ValueError("utilizador inexistente")
        finally:
            con.close()
    return t


def _ensure_user_settings_row(user_id: int) -> None:
    with _db_lock:
        con = get_connection()
        try:
            with con:
                with con.cursor() as cur:
                    cur.execute(
                        "INSERT INTO user_settings (user_id) VALUES (%s) "
                        "ON CONFLICT (user_id) DO NOTHING",
                        (int(user_id),),
                    )
        finally:
            con.close()


def get_user_model_settings(user_id: int) -> dict:
    _ensure_user_settings_row(user_id)
    with _db_lock:
        con = get_connection()
        try:
            with con.cursor() as cur:
                cur.execute(
                    """
                    SELECT system_prompt, max_new_tokens, temperature, top_p
                    FROM user_settings WHERE user_id = %s
                    """,
                    (int(user_id),),
                )
                row = cur.fetchone()
        finally:
            con.close()
    if not row:
        return {
            "system_prompt": "",
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    return {
        "system_prompt": str(row[0] or ""),
        "max_new_tokens": int(row[1]),
        "temperature": float(row[2]),
        "top_p": float(row[3]),
    }


def set_user_model_settings(
    user_id: int,
    *,
    system_prompt: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> dict:
    _ensure_user_settings_row(user_id)
    cur = get_user_model_settings(user_id)
    sp = cur["system_prompt"] if system_prompt is None else str(system_prompt)[:8000]
    ntok = cur["max_new_tokens"] if max_new_tokens is None else int(max_new_tokens)
    temp = cur["temperature"] if temperature is None else float(temperature)
    tp = cur["top_p"] if top_p is None else float(top_p)
    ntok = max(16, min(4096, ntok))
    temp = max(0.01, min(2.0, temp))
    tp = max(0.05, min(1.0, tp))
    with _db_lock:
        con = get_connection()
        try:
            with con:
                with con.cursor() as cur2:
                    cur2.execute(
                        """
                        UPDATE user_settings
                        SET system_prompt = %s, max_new_tokens = %s, temperature = %s, top_p = %s
                        WHERE user_id = %s
                        """,
                        (sp, ntok, temp, tp, int(user_id)),
                    )
        finally:
            con.close()
    return get_user_model_settings(user_id)


def is_user_admin(user_id: int) -> bool:
    with _db_lock:
        con = get_connection()
        try:
            with con.cursor() as cur:
                cur.execute("SELECT is_admin FROM users WHERE id = %s", (int(user_id),))
                row = cur.fetchone()
        finally:
            con.close()
    if not row:
        return False
    return int(row[0] or 0) == 1


def list_all_users() -> list[dict[str, Any]]:
    """Lista todos os utilizadores (painel de admin)."""
    with _db_lock:
        con = get_connection()
        try:
            with con.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, username,
                        COALESCE(NULLIF(TRIM(display_name), ''), username),
                        is_admin
                    FROM users
                    ORDER BY id
                    """
                )
                rows = cur.fetchall()
        finally:
            con.close()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": int(r[0]),
                "username": str(r[1] or ""),
                "display_name": str(r[2] or r[1] or ""),
                "is_admin": int(r[3] or 0) == 1,
            }
        )
    return out


def get_global_system_prompt() -> str:
    with _db_lock:
        con = get_connection()
        try:
            with con.cursor() as cur:
                cur.execute("SELECT global_system_prompt FROM app_global WHERE id = 1")
                row = cur.fetchone()
        finally:
            con.close()
    if not row:
        return ""
    return str(row[0] or "")


def set_global_system_prompt(text: str) -> str:
    t = str(text or "")[:8000]
    with _db_lock:
        con = get_connection()
        try:
            with con:
                with con.cursor() as cur:
                    cur.execute(
                        "UPDATE app_global SET global_system_prompt = %s WHERE id = 1",
                        (t,),
                    )
        finally:
            con.close()
    return t


def set_user_system_prompt_only(user_id: int, system_prompt: str) -> dict:
    _ensure_user_settings_row(user_id)
    sp = str(system_prompt or "")[:8000]
    with _db_lock:
        con = get_connection()
        try:
            with con:
                with con.cursor() as cur:
                    cur.execute(
                        "UPDATE user_settings SET system_prompt = %s WHERE user_id = %s",
                        (sp, int(user_id)),
                    )
        finally:
            con.close()
    return get_user_model_settings(user_id)


def ensure_bootstrap_admin() -> None:
    """
    Cria o utilizador admin ou, se já existir, alinha a palavra-passe e is_admin.
    ORACULO_NO_BOOTSTRAP_ADMIN_PASSWORD=1 — só promove a admin, não mexe na password.
    """
    sync_pwd = os.environ.get("ORACULO_NO_BOOTSTRAP_ADMIN_PASSWORD", "").strip() not in (
        "1",
        "true",
        "yes",
    )
    h = _hash_password(_BOOTSTRAP_ADMIN_PASSWORD)
    with _db_lock:
        con = get_connection()
        try:
            with con:
                with con.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM users WHERE LOWER(username) = LOWER(%s)",
                        (_BOOTSTRAP_ADMIN_USERNAME,),
                    )
                    row = cur.fetchone()
                    if row is None:
                        cur.execute(
                            """
                            INSERT INTO users (username, password_hash, is_admin)
                            VALUES (%s, %s, 1)
                            """,
                            (_BOOTSTRAP_ADMIN_USERNAME, h),
                        )
                    else:
                        uid = int(row[0])
                        if sync_pwd:
                            cur.execute(
                                "UPDATE users SET is_admin = 1, password_hash = %s WHERE id = %s",
                                (h, uid),
                            )
                        else:
                            cur.execute(
                                "UPDATE users SET is_admin = 1 WHERE id = %s",
                                (uid,),
                            )
        finally:
            con.close()
