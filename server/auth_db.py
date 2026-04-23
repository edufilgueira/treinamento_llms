"""
Utilizadores em SQLite: registo e password (bcrypt + SHA-256 de pré-hash).
Ficheiro: server/data/oraculo_users.db
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

import bcrypt

_db_lock = threading.Lock()
_SERVER_DIR = Path(__file__).resolve().parent
_DB_PATH = _SERVER_DIR / "data" / "oraculo_users.db"


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


def init_db() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db_lock:
        con = sqlite3.connect(_DB_PATH)
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
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
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
        con = sqlite3.connect(_DB_PATH)
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
