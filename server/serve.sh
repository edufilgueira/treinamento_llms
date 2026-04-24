#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Cria o venv se não existir
if [ ! -d ".venv" ]; then
    echo "Criando ambiente virtual..."
    python3 -m venv .venv
fi

# Ativa o venv
source .venv/bin/activate

# Instala dependências do servidor se faltar qualquer pacote crítico (API + modelo + DB)
if ! python3 -c "import torch, transformers, peft, accelerate, fastapi, uvicorn, bcrypt, psycopg2, dotenv, sentencepiece, tiktoken" 2>/dev/null; then
    echo "Instalando dependências do servidor..."
    pip install -r server/requirements.txt
fi

exec python3 "$SCRIPT_DIR/serve_lora.py" "$@"
