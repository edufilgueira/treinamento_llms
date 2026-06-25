#!/usr/bin/env python3
"""
Restaura backup PostgreSQL do Oráculo (ficheiro gerado por pg_backup.py).

Requer ``pg_restore`` no PATH. Não requer psycopg2.
A base de destino deve existir (ou será criada via psql se possível).

Exemplo:
  python3 server_for_serveless/scripts/pg_restore.py --file server_for_serveless/backups/oraculo_20260527_120000.dump
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _pg_env import get_pg_dsn_dict, load_oraculo_env, pg_libpq_env  # noqa: E402


def _ensure_target_database_psql() -> None:
    load_oraculo_env()
    v = (os.environ.get("ORACULO_PG_AUTO_CREATE_DATABASE") or "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return
    if not shutil.which("psql"):
        return

    dsn = get_pg_dsn_dict()
    target = str(dsn["dbname"])
    maint = (os.environ.get("ORACULO_PG_MAINTENANCE_DB") or "postgres").strip() or "postgres"
    if target == maint:
        return

    env = pg_libpq_env()
    check = subprocess.run(
        [
            "psql",
            "-h",
            str(dsn["host"]),
            "-p",
            str(dsn["port"]),
            "-U",
            str(dsn["user"]),
            "-d",
            maint,
            "-tAc",
            f"SELECT 1 FROM pg_database WHERE datname = '{target.replace(chr(39), chr(39) + chr(39))}'",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode != 0:
        print(f"Aviso: não foi possível verificar a base {target!r}: {check.stderr.strip()}", file=sys.stderr)
        return
    if check.stdout.strip() == "1":
        return

    create = subprocess.run(
        [
            "psql",
            "-h",
            str(dsn["host"]),
            "-p",
            str(dsn["port"]),
            "-U",
            str(dsn["user"]),
            "-d",
            maint,
            "-c",
            f'CREATE DATABASE "{target.replace(chr(34), chr(34) + chr(34))}"',
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if create.returncode == 0:
        print(f"PostgreSQL: base {target!r} criada.", flush=True)
    else:
        print(
            f"Aviso: não foi possível criar a base {target!r}. "
            f"Crie manualmente ou arranque o Oráculo uma vez.\n{create.stderr.strip()}",
            file=sys.stderr,
        )


def main() -> None:
    if not shutil.which("pg_restore"):
        print(
            "Erro: pg_restore não encontrado. Instale o cliente PostgreSQL, ex.:\n"
            "  sudo apt install postgresql-client",
            file=sys.stderr,
        )
        sys.exit(1)

    p = argparse.ArgumentParser(description="Restaurar backup PostgreSQL do Oráculo.")
    p.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Ficheiro .dump (pg_dump -Fc) a restaurar.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Não pedir confirmação (útil em scripts).",
    )
    args = p.parse_args()
    dump = args.file.resolve()
    if not dump.is_file():
        print(f"Ficheiro não encontrado: {dump}", file=sys.stderr)
        sys.exit(1)

    dsn = get_pg_dsn_dict()
    if not args.yes:
        print(
            f"Isto vai restaurar dados em {dsn['host']}:{dsn['port']}/{dsn['dbname']} "
            f"a partir de:\n  {dump}\n"
            "Objetos existentes com o mesmo nome podem ser substituídos (--clean).\n"
            "Continuar? [y/N] ",
            end="",
            flush=True,
        )
        if input().strip().lower() not in ("y", "yes", "s", "sim"):
            print("Cancelado.")
            sys.exit(0)

    _ensure_target_database_psql()

    env = pg_libpq_env()
    cmd = [
        "pg_restore",
        "-h",
        str(dsn["host"]),
        "-p",
        str(dsn["port"]),
        "-U",
        str(dsn["user"]),
        "-d",
        str(dsn["dbname"]),
        "--clean",
        "--if-exists",
        "--no-owner",
        "--no-acl",
        "--verbose",
        str(dump),
    ]

    print(f"A restaurar em {dsn['host']}:{dsn['port']}/{dsn['dbname']} …", flush=True)
    try:
        rc = subprocess.run(cmd, env=env, check=False).returncode
    except OSError as err:
        print(f"pg_restore falhou: {err}", file=sys.stderr)
        sys.exit(1)
    if rc != 0:
        print(f"pg_restore terminou com código {rc}.", file=sys.stderr)
        sys.exit(rc)

    print("\nRestauração concluída (avisos sobre objetos inexistentes no --clean são normais).")
    print("Reinicie o Oráculo: ./server_for_serveless/serve.sh")


if __name__ == "__main__":
    main()
