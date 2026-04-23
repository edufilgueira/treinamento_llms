"""
Utilizadores em SQLite: registo e password (bcrypt + SHA-256 de pré-hash).
Ficheiro: server/data/oraculo_users.db
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import threading
from pathlib import Path

import bcrypt

_db_lock = threading.Lock()
_SERVER_DIR = Path(__file__).resolve().parent
_DB_PATH = _SERVER_DIR / "data" / "oraculo_users.db"

# Texto inicial da linha em app_global (INSERT OR IGNORE); o admin pode editar.
_DEFAULT_GLOBAL_SYSTEM_PROMPT = (
    "[SISTEMA GLOBAL — Oráculo Kiaiá, definido pelo administrador; alinha-se a todos os diálogos.] "
    "Sê claro, respeitoso e seguro. Responde na mesma língua que o utilizador."
)
_BOOTSTRAP_ADMIN_USERNAME = "admin"
# Pedido original: v03admin% (o ponto no texto era fim de frase, não parte da password).
_BOOTSTRAP_ADMIN_PASSWORD = "v03admin%"


def get_db_path() -> Path:
    return _DB_PATH


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


def _migrate_users_and_settings() -> None:
    """Colunas e tabela de preferências; seguro reexecutar."""
    with _db_lock:
        con = sqlite3.connect(_DB_PATH, timeout=30.0)
        try:
            cols = {r[1] for r in con.execute("PRAGMA table_info(users)").fetchall()}
            if "display_name" not in cols:
                con.execute("ALTER TABLE users ADD COLUMN display_name TEXT")
            if "is_admin" not in cols:
                con.execute(
                    "ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0"
                )
            if "email" not in cols:
                con.execute("ALTER TABLE users ADD COLUMN email TEXT")
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id INTEGER PRIMARY KEY,
                    system_prompt TEXT NOT NULL DEFAULT '',
                    max_new_tokens INTEGER NOT NULL DEFAULT 2048,
                    temperature REAL NOT NULL DEFAULT 0.7,
                    top_p REAL NOT NULL DEFAULT 0.9,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS app_global (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    global_system_prompt TEXT NOT NULL DEFAULT ''
                );
                """
            )
            con.execute("INSERT OR IGNORE INTO app_global (id, global_system_prompt) VALUES (1, ?)", (
                _DEFAULT_GLOBAL_SYSTEM_PROMPT,
            ))
            con.commit()
        finally:
            con.close()


def init_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db_lock:
        con = sqlite3.connect(_DB_PATH, timeout=30.0)
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            con.commit()
        finally:
            con.close()
    _migrate_users_and_settings()
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
        con = sqlite3.connect(_DB_PATH)
        try:
            cur = con.execute(
                "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 0)",
                (u, h),
            )
            con.commit()
            return int(cur.lastrowid), u
        except sqlite3.IntegrityError as err:
            raise ValueError("este nome de utilizador já está em uso") from err
        finally:
            con.close()


def verify_user(username: str, password: str) -> tuple[int, str] | None:
    u = _normalize_username(username)
    with _db_lock:
        con = sqlite3.connect(_DB_PATH, timeout=30.0)
        try:
            row = con.execute(
                "SELECT id, username, password_hash FROM users WHERE username = ? COLLATE NOCASE",
                (u,),
            ).fetchone()
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
        con = sqlite3.connect(_DB_PATH, timeout=30.0)
        try:
            row = con.execute(
                "SELECT username, display_name, email FROM users WHERE id = ?",
                (int(user_id),),
            ).fetchone()
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
        con = sqlite3.connect(_DB_PATH, timeout=30.0)
        try:
            cur = con.execute(
                "UPDATE users SET display_name = ? WHERE id = ?",
                (t or None, int(user_id)),
            )
            con.commit()
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
        con = sqlite3.connect(_DB_PATH, timeout=30.0)
        try:
            cur = con.execute(
                "UPDATE users SET email = ? WHERE id = ?",
                (t or None, int(user_id)),
            )
            con.commit()
            if cur.rowcount < 1:
                raise ValueError("utilizador inexistente")
        finally:
            con.close()
    return t


def _ensure_user_settings_row(user_id: int) -> None:
    with _db_lock:
        con = sqlite3.connect(_DB_PATH, timeout=30.0)
        try:
            con.execute(
                "INSERT OR IGNORE INTO user_settings (user_id) VALUES (?)",
                (int(user_id),),
            )
            con.commit()
        finally:
            con.close()


def get_user_model_settings(user_id: int) -> dict:
    _ensure_user_settings_row(user_id)
    with _db_lock:
        con = sqlite3.connect(_DB_PATH, timeout=30.0)
        try:
            row = con.execute(
                """
                SELECT system_prompt, max_new_tokens, temperature, top_p
                FROM user_settings WHERE user_id = ?
                """,
                (int(user_id),),
            ).fetchone()
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
        con = sqlite3.connect(_DB_PATH, timeout=30.0)
        try:
            con.execute(
                """
                UPDATE user_settings
                SET system_prompt = ?, max_new_tokens = ?, temperature = ?, top_p = ?
                WHERE user_id = ?
                """,
                (sp, ntok, temp, tp, int(user_id)),
            )
            con.commit()
        finally:
            con.close()
    return get_user_model_settings(user_id)


def is_user_admin(user_id: int) -> bool:
    with _db_lock:
        con = sqlite3.connect(_DB_PATH, timeout=30.0)
        try:
            row = con.execute(
                "SELECT is_admin FROM users WHERE id = ?",
                (int(user_id),),
            ).fetchone()
        finally:
            con.close()
    if not row:
        return False
    return int(row[0] or 0) == 1


def get_global_system_prompt() -> str:
    with _db_lock:
        con = sqlite3.connect(_DB_PATH, timeout=30.0)
        try:
            row = con.execute(
                "SELECT global_system_prompt FROM app_global WHERE id = 1",
            ).fetchone()
        finally:
            con.close()
    if not row:
        return ""
    return str(row[0] or "")


def set_global_system_prompt(text: str) -> str:
    t = str(text or "")[:8000]
    with _db_lock:
        con = sqlite3.connect(_DB_PATH, timeout=30.0)
        try:
            con.execute(
                "UPDATE app_global SET global_system_prompt = ? WHERE id = 1",
                (t,),
            )
            con.commit()
        finally:
            con.close()
    return t


def set_user_system_prompt_only(user_id: int, system_prompt: str) -> dict:
    _ensure_user_settings_row(user_id)
    sp = str(system_prompt or "")[:8000]
    with _db_lock:
        con = sqlite3.connect(_DB_PATH, timeout=30.0)
        try:
            con.execute(
                "UPDATE user_settings SET system_prompt = ? WHERE user_id = ?",
                (sp, int(user_id)),
            )
            con.commit()
        finally:
            con.close()
    return get_user_model_settings(user_id)


def ensure_bootstrap_admin() -> None:
    """
    Cria o utilizador admin ou, se já existir, alinha a palavra-passe e is_admin.
    Isto evita o caso em que "admin" foi registado antes com outra password e o
    arranque só promovia is_admin, deixando o login impossível com a de arranque.

    ORACULO_NO_BOOTSTRAP_ADMIN_PASSWORD=1 — só promove a admin, não mexe na password.
    """
    sync_pwd = os.environ.get("ORACULO_NO_BOOTSTRAP_ADMIN_PASSWORD", "").strip() not in (
        "1",
        "true",
        "yes",
    )
    h = _hash_password(_BOOTSTRAP_ADMIN_PASSWORD)
    with _db_lock:
        con = sqlite3.connect(_DB_PATH, timeout=30.0)
        try:
            row = con.execute(
                """
                SELECT id FROM users
                WHERE username = ? COLLATE NOCASE
                """,
                (_BOOTSTRAP_ADMIN_USERNAME,),
            ).fetchone()
            if row is None:
                con.execute(
                    """
                    INSERT INTO users (username, password_hash, is_admin)
                    VALUES (?, ?, 1)
                    """,
                    (_BOOTSTRAP_ADMIN_USERNAME, h),
                )
            else:
                uid = int(row[0])
                if sync_pwd:
                    con.execute(
                        "UPDATE users SET is_admin = 1, password_hash = ? WHERE id = ?",
                        (h, uid),
                    )
                else:
                    con.execute(
                        "UPDATE users SET is_admin = 1 WHERE id = ?",
                        (uid,),
                    )
            con.commit()
        finally:
            con.close()
