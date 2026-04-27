#!/usr/bin/env python3
"""
Ponto de entrada: configura o PYTHONPATH, carrega `.env` e arranca o Uvicorn.

  cd <raiz do projeto> && python3 server/serve_lora.py
  # ou: ./server/serve.sh

Inferência: llama-server (URL no admin ou .env), GGUF local, ou pasta **modelo fundido** (HF).
Não há carregamento base+LoRA em runtime — usa sempre merge completo para PyTorch.
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

from server.bootstrap import load_dotenv_early

load_dotenv_early()

from data_config import apply_loading_progress_env

apply_loading_progress_env()

if not os.environ.get("PYTORCH_ALLOC_CONF"):
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def main() -> None:
    from server.main import main as run_app

    run_app()


if __name__ == "__main__":
    main()
