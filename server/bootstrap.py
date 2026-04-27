"""Carrega `.env` antes de importar o resto da aplicação."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def load_dotenv_early() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as err:
        print(
            "Erro: falta o pacote python-dotenv. Instale com:  pip install python-dotenv",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from err
    server_dir = Path(__file__).resolve().parent
    root = server_dir.parent
    load_dotenv(root / ".env")
    load_dotenv(server_dir / ".env", override=True)
    root_env = root / ".env"
    server_env = server_dir / ".env"
    if not root_env.is_file() and not server_env.is_file():
        print(
            "Erro: é obrigatório existir um ficheiro `.env`.\n"
            f"  Cria `{server_env}` (recomendado) ou `{root_env}` — por exemplo:\n"
            f"    cp {server_dir / '.env.example'} {server_env}\n"
            "  Depois edita e define pelo menos ORACULO_PG_HOST (e as restantes chaves PostgreSQL).",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
    if not (os.environ.get("ORACULO_PG_HOST") or "").strip():
        print(
            "Erro: ORACULO_PG_HOST não está definido. Adiciona-o ao `.env` (ver server/.env.example).",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
