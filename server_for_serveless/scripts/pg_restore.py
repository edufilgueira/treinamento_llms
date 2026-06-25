#!/usr/bin/env python3
"""
Restaura backup PostgreSQL do Oráculo (ficheiro gerado por pg_backup.py).

Requer ``pg_restore`` no PATH. A base de destino deve existir (ou use
ORACULO_PG_AUTO_CREATE_DATABASE=1 e arranque o servidor uma vez).

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

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from server_for_serveless.db.pg_db import ensure_target_database, get_pg_dsn_dict  # noqa: E402


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

    try:
        ensure_target_database()
    except Exception as err:
        print(f"Aviso ao garantir base: {err}", file=sys.stderr)

    env = os.environ.copy()
    if dsn.get("password"):
        env["PGPASSWORD"] = str(dsn["password"])
    if dsn.get("sslmode"):
        env.setdefault("PGSSLMODE", str(dsn["sslmode"]))

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
