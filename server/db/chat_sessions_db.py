"""
Sessões de chat por utilizador (PostgreSQL, mesma base que auth).
"""

from __future__ import annotations

import re
import threading
from typing import Any

from .pg_db import get_connection, init_schema

_db_lock = threading.Lock()

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


def init_chat_tables() -> None:
    """Garante o esquema; idempotente (já feito em auth_db.init_db; mantido para compat)."""
    with _db_lock:
        init_schema()


def create_session(user_id: int) -> int:
    with _db_lock:
        con = get_connection()
        try:
            with con:
                with con.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO chat_sessions (user_id, title, title_done)
                        VALUES (%s, %s, 0)
                        RETURNING id
                        """,
                        (user_id, DEFAULT_SESSION_TITLE),
                    )
                    row = cur.fetchone()
                    if not row:
                        raise RuntimeError("falha insert chat_sessions")
                    return int(row[0])
        finally:
            con.close()


def _owns(user_id: int, session_id: int) -> bool:
    with _db_lock:
        con = get_connection()
        try:
            with con.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM chat_sessions WHERE id = %s AND user_id = %s",
                    (session_id, user_id),
                )
                return cur.fetchone() is not None
        finally:
            con.close()


def list_sessions(user_id: int, limit: int = 100) -> list[dict[str, Any]]:
    with _db_lock:
        con = get_connection()
        try:
            with con.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, title, created_at, updated_at,
                           total_output_tokens, total_gen_seconds
                    FROM chat_sessions
                    WHERE user_id = %s
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (user_id, limit),
                )
                rows = cur.fetchall()
        finally:
            con.close()
    out: list[dict[str, Any]] = []
    for r in rows:
        ca, ua = r[2], r[3]
        tot_tok, tot_sec = int(r[4] or 0), float(r[5] or 0.0)
        out.append(
            {
                "id": r[0],
                "title": r[1],
                "created_at": ca.isoformat() if hasattr(ca, "isoformat") else str(ca or ""),
                "updated_at": ua.isoformat() if hasattr(ua, "isoformat") else str(ua or ""),
                "total_output_tokens": tot_tok,
                "total_gen_seconds": tot_sec,
            }
        )
    return out


def get_session_messages(
    user_id: int,
    session_id: int,
    *,
    include_message_admin_meta: bool = False,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]] | None:
    if not _owns(user_id, session_id):
        return None
    with _db_lock:
        con = get_connection()
        try:
            with con.cursor() as cur:
                cur.execute(
                    """
                    SELECT title, total_output_tokens, total_gen_seconds
                    FROM chat_sessions WHERE id = %s
                    """,
                    (session_id,),
                )
                trow = cur.fetchone()
                if not trow:
                    return None
                title = str(trow[0])
                tot_tok = int(trow[1] or 0)
                tot_sec = float(trow[2] or 0.0)
                if include_message_admin_meta:
                    cur.execute(
                        """
                        SELECT id, role, content, output_tokens, gen_seconds,
                               tokens_per_sec, COALESCE(admin_reviewed, 0)
                        FROM chat_messages
                        WHERE session_id = %s
                        ORDER BY pos ASC, id ASC
                        """,
                        (session_id,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT role, content, output_tokens, gen_seconds, tokens_per_sec
                        FROM chat_messages
                        WHERE session_id = %s
                        ORDER BY pos ASC, id ASC
                        """,
                        (session_id,),
                    )
                rows = cur.fetchall()
        finally:
            con.close()
    session_meta: dict[str, Any] = {
        "total_output_tokens": tot_tok,
        "total_gen_seconds": tot_sec,
    }
    msg_list: list[dict[str, Any]] = []
    if include_message_admin_meta:
        for r in rows:
            role = str(r[1])
            d: dict[str, Any] = {
                "id": int(r[0]),
                "role": role,
                "content": str(r[2]),
                "admin_reviewed": bool(int(r[6] or 0)),
            }
            if role == "assistant":
                if r[3] is not None:
                    d["output_tokens"] = int(r[3])
                if r[4] is not None:
                    d["gen_seconds"] = float(r[4])
                if r[5] is not None:
                    d["tokens_per_sec"] = float(r[5])
            msg_list.append(d)
    else:
        for r in rows:
            d = {"role": str(r[0]), "content": str(r[1])}
            if str(r[0]) == "assistant":
                if r[2] is not None:
                    d["output_tokens"] = int(r[2])
                if r[3] is not None:
                    d["gen_seconds"] = float(r[3])
                if r[4] is not None:
                    d["tokens_per_sec"] = float(r[4])
            msg_list.append(d)
    return (msg_list, title, session_meta)


def set_assistant_message_admin_reviewed(
    user_id: int,
    session_id: int,
    message_id: int,
    reviewed: bool,
) -> bool:
    """Marca a mensagem assistente do turno como revista (painel admin)."""
    if not _owns(user_id, session_id) or message_id < 1:
        return False
    flag = 1 if reviewed else 0
    with _db_lock:
        con = get_connection()
        try:
            with con:
                with con.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE chat_messages
                        SET admin_reviewed = %s
                        WHERE id = %s AND session_id = %s AND role = 'assistant'
                        """,
                        (flag, message_id, session_id),
                    )
                    return cur.rowcount > 0
        finally:
            con.close()


def delete_session(user_id: int, session_id: int) -> bool:
    if not _owns(user_id, session_id):
        return False
    with _db_lock:
        con = get_connection()
        try:
            with con:
                with con.cursor() as cur:
                    cur.execute(
                        "DELETE FROM chat_sessions WHERE id = %s AND user_id = %s",
                        (session_id, user_id),
                    )
                    return cur.rowcount > 0
        finally:
            con.close()


def append_turn(
    user_id: int,
    session_id: int,
    user_text: str,
    assistant_text: str,
    *,
    output_tokens: int | None = None,
    gen_seconds: float | None = None,
    tokens_per_sec: float | None = None,
) -> bool:
    if not _owns(user_id, session_id):
        return False
    user_text = user_text or ""
    assistant_text = assistant_text or ""
    add_tok = int(output_tokens) if output_tokens is not None else 0
    add_sec = float(gen_seconds) if gen_seconds is not None else 0.0
    with _db_lock:
        con = get_connection()
        try:
            with con:
                with con.cursor() as cur:
                    cur.execute(
                        "SELECT COALESCE(MAX(pos), -1) FROM chat_messages WHERE session_id = %s",
                        (session_id,),
                    )
                    row = cur.fetchone()
                    base = int(row[0]) + 1
                    cur.execute(
                        """
                        INSERT INTO chat_messages (
                            session_id, role, content, pos,
                            output_tokens, gen_seconds, tokens_per_sec
                        )
                        VALUES (%s, 'user', %s, %s, NULL, NULL, NULL)
                        """,
                        (session_id, user_text, base),
                    )
                    cur.execute(
                        """
                        INSERT INTO chat_messages (
                            session_id, role, content, pos,
                            output_tokens, gen_seconds, tokens_per_sec
                        )
                        VALUES (%s, 'assistant', %s, %s, %s, %s, %s)
                        """,
                        (
                            session_id,
                            assistant_text,
                            base + 1,
                            output_tokens,
                            gen_seconds,
                            tokens_per_sec,
                        ),
                    )
                    if output_tokens is not None or gen_seconds is not None:
                        cur.execute(
                            """
                            UPDATE chat_sessions
                            SET updated_at = NOW(),
                                total_output_tokens = total_output_tokens + %s,
                                total_gen_seconds = total_gen_seconds + %s
                            WHERE id = %s
                            """,
                            (add_tok, add_sec, session_id),
                        )
                    else:
                        cur.execute(
                            "UPDATE chat_sessions SET updated_at = NOW() WHERE id = %s",
                            (session_id,),
                        )
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
        con = get_connection()
        try:
            with con:
                with con.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE chat_sessions
                        SET title = %s, title_done = 1, updated_at = NOW()
                        WHERE id = %s AND user_id = %s
                        """,
                        (t, session_id, user_id),
                    )
                    return cur.rowcount > 0
        finally:
            con.close()


def set_session_title_from_model(user_id: int, session_id: int, title: str) -> bool:
    t = re.sub(r"\s+", " ", (title or "").strip())[:200]
    if not t:
        return False
    with _db_lock:
        con = get_connection()
        try:
            with con:
                with con.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE chat_sessions
                        SET title = %s, title_done = 1, updated_at = NOW()
                        WHERE id = %s AND user_id = %s AND title_done = 0
                        """,
                        (t, session_id, user_id),
                    )
                    return cur.rowcount > 0
        finally:
            con.close()


def user_owns_session(user_id: int, session_id: int) -> bool:
    return _owns(user_id, session_id)


def should_generate_title(user_id: int, session_id: int) -> bool:
    with _db_lock:
        con = get_connection()
        try:
            with con.cursor() as cur:
                cur.execute(
                    "SELECT title_done, title FROM chat_sessions WHERE id = %s AND user_id = %s",
                    (session_id, user_id),
                )
                row = cur.fetchone()
        finally:
            con.close()
    if not row:
        return False
    done, tit = int(row[0]), str(row[1])
    return not done and tit == DEFAULT_SESSION_TITLE
