"""
Lê ORACULO_PG_* do .env sem importar psycopg2 (scripts de backup/restore).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_SERVER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _SERVER_DIR.parent


def _parse_env_file(path: Path, *, override: bool) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if override or key not in os.environ:
            os.environ[key] = val


def load_oraculo_env() -> None:
    """Carrega `.env` na raiz e `server_for_serveless/.env` (este sobrepõe)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        _parse_env_file(_REPO_ROOT / ".env", override=False)
        _parse_env_file(_SERVER_DIR / ".env", override=True)
        return
    load_dotenv(_REPO_ROOT / ".env")
    load_dotenv(_SERVER_DIR / ".env", override=True)


def get_pg_dsn_dict() -> dict[str, Any]:
    load_oraculo_env()
    host = (os.environ.get("ORACULO_PG_HOST") or "").strip()
    if not host:
        raise RuntimeError(
            "ORACULO_PG_HOST não definido. Configure server_for_serveless/.env "
            "(copie de .env.example)."
        )
    dsn: dict[str, Any] = {
        "host": host,
        "port": int(os.environ.get("ORACULO_PG_PORT", "5432").strip() or "5432"),
        "user": (os.environ.get("ORACULO_PG_USER") or os.environ.get("PGUSER") or "postgres").strip()
        or "postgres",
        "password": (
            os.environ.get("ORACULO_PG_PASSWORD") or os.environ.get("PGPASSWORD") or ""
        ).strip(),
        "dbname": (
            os.environ.get("ORACULO_PG_DATABASE") or os.environ.get("PGDATABASE") or "oraculo"
        ).strip()
        or "oraculo",
    }
    ssl = (os.environ.get("ORACULO_PG_SSLMODE", "prefer") or "prefer").strip()
    if ssl:
        dsn["sslmode"] = ssl
    return dsn


def pg_libpq_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Ambiente para pg_dump / pg_restore / psql (PGPASSWORD, PGSSLMODE)."""
    dsn = get_pg_dsn_dict()
    env = dict(base or os.environ)
    if dsn.get("password"):
        env["PGPASSWORD"] = str(dsn["password"])
    if dsn.get("sslmode"):
        env["PGSSLMODE"] = str(dsn["sslmode"])
    return env
