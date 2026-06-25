#!/usr/bin/env python3
"""
Backup completo da base PostgreSQL do Oráculo (utilizadores, chats, app_global).

Usa credenciais de server_for_serveless/.env. Não requer psycopg2 — só pg_dump.
Requer ``pg_dump`` no PATH (pacote postgresql-client no Linux).

Exemplo:
  python3 server_for_serveless/scripts/pg_backup.py
  python3 server_for_serveless/scripts/pg_backup.py --out ~/backups/oraculo.dump
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _pg_env import get_pg_dsn_dict, pg_libpq_env  # noqa: E402


def _default_out_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = _SCRIPT_DIR.parent / "backups"
    base.mkdir(parents=True, exist_ok=True)
    dsn = get_pg_dsn_dict()
    return base / f"{dsn['dbname']}_{stamp}.dump"


def main() -> None:
    if not shutil.which("pg_dump"):
        print(
            "Erro: pg_dump não encontrado. Instale o cliente PostgreSQL, ex.:\n"
            "  sudo apt install postgresql-client",
            file=sys.stderr,
        )
        sys.exit(1)

    p = argparse.ArgumentParser(description="Backup PostgreSQL do Oráculo (pg_dump -Fc).")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Ficheiro .dump de saída (padrão: server_for_serveless/backups/<db>_YYYYMMDD_HHMMSS.dump)",
    )
    args = p.parse_args()
    out = args.out or _default_out_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    dsn = get_pg_dsn_dict()
    env = pg_libpq_env()

    cmd = [
        "pg_dump",
        "-h",
        str(dsn["host"]),
        "-p",
        str(dsn["port"]),
        "-U",
        str(dsn["user"]),
        "-d",
        str(dsn["dbname"]),
        "--format=custom",
        "--no-owner",
        "--no-acl",
        f"--file={out}",
    ]

    print(f"A ligar a {dsn['host']}:{dsn['port']}/{dsn['dbname']} …", flush=True)
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as err:
        print(f"pg_dump falhou (código {err.returncode}).", file=sys.stderr)
        sys.exit(err.returncode or 1)

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Backup concluído: {out.resolve()}")
    print(f"Tamanho: {size_mb:.2f} MiB")
    print("\nGuarde este ficheiro fora do servidor antes de formatar o PostgreSQL.")
    print("Restaurar: python3 server_for_serveless/scripts/pg_restore.py --file", out)


if __name__ == "__main__":
    main()
