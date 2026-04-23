"""
Sessões de chat por utilizador (SQLite, mesma DB que auth).
"""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path

_db_lock = threading.Lock()
_PATH = Path(__file__).resolve().parent / "data" / "oraculo_users.db"

DEFAULT_SESSION_TITLE = "Novo chat"


def plain_for_storage(s: str) -> str:
    """
    Reduz Markdown a texto plano (ex.: geração de título a partir de excertos).
    O conteúdo das mensagens em `append_turn` guarda-se na íntegra.
    """
    if not s or not s.strip():
        return (s or "").strip()

    t = str(s)
    t = re.sub(r"```[\w-]*\n?([\s\S]*?)```", r"\1", t)
    t = re.sub(r"~~(.+?)~~", r"\1", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)
    for _ in range(8):
        prev = t
        t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
        t = re.sub(r"__([^_]+)__", r"\1", t)
        if t == prev:
            break
    t = re.sub(r"(?m)^\s*#{1,6}\s+", "", t)
    t = re.sub(r"(?m)^\s*>\s?", "", t)
    t = re.sub(r"(?m)^\s*(?:\d+)[.)]\s+", "", t)
    t = re.sub(r"(?m)^\s*[-*+]\s+", "", t)
    t = re.sub(r"(?m)^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(_PATH, timeout=30.0)


def init_chat_tables() -> None:
    with _db_lock:
        con = _connect()
        try:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL DEFAULT 'Novo chat',
                    title_done INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
                ON chat_sessions(user_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    pos INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_chat_messages_session_pos
                ON chat_messages(session_id, pos);
                """
            )
            con.commit()
        finally:
            con.close()


def create_session(user_id: int) -> int:
    with _db_lock:
        con = _connect()
        try:
            cur = con.execute(
                """
                INSERT INTO chat_sessions (user_id, title, title_done)
                VALUES (?, ?, 0)
                """,
                (user_id, DEFAULT_SESSION_TITLE),
            )
            con.commit()
            return int(cur.lastrowid)
        finally:
            con.close()


def _owns(user_id: int, session_id: int) -> bool:
    with _db_lock:
        con = _connect()
        try:
            row = con.execute(
                "SELECT 1 FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
            return bool(row)
        finally:
            con.close()


def list_sessions(user_id: int, limit: int = 100) -> list[dict]:
    with _db_lock:
        con = _connect()
        try:
            rows = con.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM chat_sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        finally:
            con.close()
    return [
        {
            "id": r[0],
            "title": r[1],
            "created_at": r[2],
            "updated_at": r[3],
        }
        for r in rows
    ]


def get_session_messages(user_id: int, session_id: int) -> tuple[list[dict], str] | None:
    if not _owns(user_id, session_id):
        return None
    with _db_lock:
        con = _connect()
        try:
            trow = con.execute(
                "SELECT title FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if not trow:
                return None
            title = str(trow[0])
            rows = con.execute(
                """
                SELECT role, content FROM chat_messages
                WHERE session_id = ?
                ORDER BY pos ASC, id ASC
                """,
                (session_id,),
            ).fetchall()
        finally:
            con.close()
    return (
        [{"role": str(r[0]), "content": str(r[1])} for r in rows],
        title,
    )


def delete_session(user_id: int, session_id: int) -> bool:
    if not _owns(user_id, session_id):
        return False
    with _db_lock:
        con = _connect()
        try:
            con.execute("DELETE FROM chat_sessions WHERE id = ? AND user_id = ?", (session_id, user_id))
            con.commit()
            return True
        finally:
            con.close()


def append_turn(
    user_id: int, session_id: int, user_text: str, assistant_text: str
) -> bool:
    if not _owns(user_id, session_id):
        return False
    user_text = user_text or ""
    assistant_text = assistant_text or ""
    with _db_lock:
        con = _connect()
        try:
            row = con.execute(
                "SELECT COALESCE(MAX(pos), -1) FROM chat_messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            base = int(row[0]) + 1
            con.execute(
                "INSERT INTO chat_messages (session_id, role, content, pos) VALUES (?, 'user', ?, ?)",
                (session_id, user_text, base),
            )
            con.execute(
                "INSERT INTO chat_messages (session_id, role, content, pos) VALUES (?, 'assistant', ?, ?)",
                (session_id, assistant_text, base + 1),
            )
            con.execute(
                "UPDATE chat_sessions SET updated_at = datetime('now') WHERE id = ?",
                (session_id,),
            )
            con.commit()
        finally:
            con.close()
    return True


def set_session_title_user(user_id: int, session_id: int, title: str) -> bool:
    """Atualiza o título a pedido do utilizador (renomear na UI)."""
    t = re.sub(r"\s+", " ", (title or "").strip())[:200]
    if not t:
        return False
    if not _owns(user_id, session_id):
        return False
    with _db_lock:
        con = _connect()
        try:
            cur = con.execute(
                """
                UPDATE chat_sessions
                SET title = ?, title_done = 1, updated_at = datetime('now')
                WHERE id = ? AND user_id = ?
                """,
                (t, session_id, user_id),
            )
            con.commit()
            return cur.rowcount > 0
        finally:
            con.close()


def set_session_title_from_model(user_id: int, session_id: int, title: str) -> bool:
    t = re.sub(r"\s+", " ", (title or "").strip())[:200]
    if not t:
        return False
    with _db_lock:
        con = _connect()
        try:
            cur = con.execute(
                """
                UPDATE chat_sessions
                SET title = ?, title_done = 1, updated_at = datetime('now')
                WHERE id = ? AND user_id = ? AND title_done = 0
                """,
                (t, session_id, user_id),
            )
            con.commit()
            return cur.rowcount > 0
        finally:
            con.close()


def user_owns_session(user_id: int, session_id: int) -> bool:
    return _owns(user_id, session_id)


def should_generate_title(user_id: int, session_id: int) -> bool:
    with _db_lock:
        con = _connect()
        try:
            row = con.execute(
                "SELECT title_done, title FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
        finally:
            con.close()
    if not row:
        return False
    done, tit = int(row[0]), str(row[1])
    return not done and tit == DEFAULT_SESSION_TITLE
