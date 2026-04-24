"""
Ligação PostgreSQL e criação de tabelas (utilizadores, sessões, mensagens).
Configuração por variáveis de ambiente (podes definir no `.env` na raiz do
projecto ou em `server/.env`; o segundo substitui o primeiro) ou `export` no shell:

  ORACULO_PG_HOST   (padrão: 187.77.44.167 — confere 187 vs 87, é erro comum)
  ORACULO_PG_PORT   (padrão: 5432)
  ORACULO_PG_USER
  ORACULO_PG_PASSWORD
  ORACULO_PG_DATABASE  (padrão: oraculo)
  ORACULO_PG_SSLMODE   (padrão: prefer)
  ORACULO_PG_CONNECT_TIMEOUT  segundos para ligação (padrão: 15; evita bloquear à espera do PG)
  ORACULO_PG_MAINTENANCE_DB  base à qual ligar para CREATE DATABASE (padrão: postgres)
  ORACULO_PG_AUTO_CREATE_DATABASE  1/0 — criar ORACULO_PG_DATABASE se não existir (padrão: 1)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2 import sql


def _load_dotenv_if_available() -> None:
    """Carrega `.env` na raiz do projecto e opcionalmente `server/.env` (este sobrepõe)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    _server_dir = Path(__file__).resolve().parent
    _root = _server_dir.parent
    load_dotenv(_root / ".env")
    load_dotenv(_server_dir / ".env", override=True)


_load_dotenv_if_available()

# Evitar import circular com auth_db; alinhar com auth_db._DEFAULT_GLOBAL_SYSTEM_PROMPT
_DEFAULT_GLOBAL = (
    "[SISTEMA GLOBAL — Oráculo Kiaiá, definido pelo administrador; alinha-se a todos os diálogos.] "
    "Sê claro, respeitoso e seguro. Responde na mesma língua que o utilizador."
)

_pg_params: dict[str, Any] | None = None


def get_pg_dsn_dict() -> dict[str, Any]:
    global _pg_params
    if _pg_params is not None:
        return dict(_pg_params)
    _pg_params = {
        "host": os.environ.get("ORACULO_PG_HOST", "187.77.44.167").strip() or "187.77.44.167",
        "port": int(os.environ.get("ORACULO_PG_PORT", "5432").strip() or "5432"),
        "user": (os.environ.get("ORACULO_PG_USER") or os.environ.get("PGUSER") or "").strip() or "postgres",
        "password": (os.environ.get("ORACULO_PG_PASSWORD") or os.environ.get("PGPASSWORD") or "").strip(),
        "dbname": (os.environ.get("ORACULO_PG_DATABASE") or os.environ.get("PGDATABASE") or "oraculo").strip()
        or "oraculo",
    }
    ssl = (os.environ.get("ORACULO_PG_SSLMODE", "prefer") or "prefer").strip()
    if ssl:
        _pg_params["sslmode"] = ssl
    try:
        _ct = int((os.environ.get("ORACULO_PG_CONNECT_TIMEOUT") or "15").strip() or "15")
    except ValueError:
        _ct = 15
    if _ct > 0:
        _pg_params["connect_timeout"] = _ct
    return dict(_pg_params)


def get_connection() -> Any:
    return psycopg2.connect(**get_pg_dsn_dict())


def ensure_target_database() -> bool:
    """
    Garante que a base `dbname` do DSN existe, criando-a a partir de
    `ORACULO_PG_MAINTENANCE_DB` (por defeito `postgres`) com privilégio adequado.
    Retorna True se tiver criado a base nesta chamada.
    """
    dsn = get_pg_dsn_dict()
    target = (dsn.get("dbname") or "oraculo").strip() or "oraculo"
    maint = (os.environ.get("ORACULO_PG_MAINTENANCE_DB") or "postgres").strip() or "postgres"
    v = (os.environ.get("ORACULO_PG_AUTO_CREATE_DATABASE") or "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if target == maint:
        return False
    params = dict(dsn)
    params["dbname"] = maint
    con = psycopg2.connect(**params)
    con.autocommit = True
    try:
        with con.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target,))
            if cur.fetchone() is not None:
                return False
        with con.cursor() as cur:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target)))
    except psycopg2.Error as e:
        raise RuntimeError(
            f"Não foi possível criar a base de dados {target!r} (liguei a {maint!r} para o efeito). "
            f"Cria manualmente no servidor, como utilizador com permissão, por exemplo: "
            f"createdb {target}   ou  CREATE DATABASE {target};\n"
            f"Erro original: {e}"
        ) from e
    finally:
        con.close()
    return True


def init_schema() -> None:
    """
    Cria tabelas e índices (idempotente). Chamado no arranque do servidor.
    """
    dsn0 = get_pg_dsn_dict()
    target_name = dsn0.get("dbname", "oraculo")
    if ensure_target_database():
        print(
            f"PostgreSQL: a base {target_name!r} foi criada (antes não existia).",
            flush=True,
        )
    con = get_connection()
    con.autocommit = True
    try:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT,
                email TEXT,
                is_admin SMALLINT NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_lower
            ON users (LOWER(username))
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY
                    REFERENCES users (id) ON DELETE CASCADE,
                system_prompt TEXT NOT NULL DEFAULT '',
                max_new_tokens INTEGER NOT NULL DEFAULT 2048,
                temperature DOUBLE PRECISION NOT NULL DEFAULT 0.7,
                top_p DOUBLE PRECISION NOT NULL DEFAULT 0.9
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS app_global (
                id INTEGER PRIMARY KEY,
                global_system_prompt TEXT NOT NULL DEFAULT ''
            )
            """
        )
        cur.execute(
            "INSERT INTO app_global (id, global_system_prompt) VALUES (1, %s) "
            "ON CONFLICT (id) DO NOTHING",
            (_DEFAULT_GLOBAL,),
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL
                    REFERENCES users (id) ON DELETE CASCADE,
                title TEXT NOT NULL DEFAULT 'Novo chat',
                title_done SMALLINT NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated
            ON chat_sessions (user_id, updated_at DESC)
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id SERIAL PRIMARY KEY,
                session_id INTEGER NOT NULL
                    REFERENCES chat_sessions (id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                pos INTEGER NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_pos
            ON chat_messages (session_id, pos)
            """
        )
        _ensure_chat_stats_columns(cur)
    finally:
        con.close()


def _ensure_chat_stats_columns(cur: Any) -> None:
    """Migração idempotente: totais da sessão e métricas por mensagem (assistente)."""
    cur.execute(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'chat_sessions'
        """
    )
    have_s = {r[0] for r in cur.fetchall()}
    if "total_output_tokens" not in have_s:
        cur.execute(
            """
            ALTER TABLE chat_sessions
            ADD COLUMN total_output_tokens INTEGER NOT NULL DEFAULT 0
            """
        )
    if "total_gen_seconds" not in have_s:
        cur.execute(
            """
            ALTER TABLE chat_sessions
            ADD COLUMN total_gen_seconds DOUBLE PRECISION NOT NULL DEFAULT 0
            """
        )
    cur.execute(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'chat_messages'
        """
    )
    have_m = {r[0] for r in cur.fetchall()}
    if "output_tokens" not in have_m:
        cur.execute("ALTER TABLE chat_messages ADD COLUMN output_tokens INTEGER NULL")
    if "gen_seconds" not in have_m:
        cur.execute(
            "ALTER TABLE chat_messages ADD COLUMN gen_seconds DOUBLE PRECISION NULL"
        )
    if "tokens_per_sec" not in have_m:
        cur.execute(
            "ALTER TABLE chat_messages ADD COLUMN tokens_per_sec DOUBLE PRECISION NULL"
        )
