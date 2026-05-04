#!/usr/bin/env python3
"""
Ponto de entrada: configura o PYTHONPATH, carrega `.env` e arranca o Uvicorn.

  cd <raiz do projeto> && python3 server_for_serveless/serve_lora.py
  # ou: ./server_for_serveless/serve.sh

Inferência: llama-server HTTP (URL no admin ou ORACULO_LLAMA_CPP_BASE_URL) ou Runpod Serverless
(ORACULO_RUNPOD_ENDPOINT_ID + ORACULO_RUNPOD_API_KEY). Este processo não carrega pesos localmente.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SERVER_DIR = Path(__file__).resolve().parent
for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "trein"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from server_for_serveless.bootstrap import load_dotenv_early

load_dotenv_early()


def main() -> None:
    from server_for_serveless.main import main as run_app

    run_app()


if __name__ == "__main__":
    main()
