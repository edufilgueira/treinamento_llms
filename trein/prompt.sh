#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Uso: ./trein/prompt.sh "seu prompt aqui"

if [ -z "$1" ]; then
    echo "Uso: ./trein/prompt.sh \"seu prompt aqui\""
    exit 1
fi

# Ativa o venv
source .venv/bin/activate

# Roda a inferência (a partir da raiz do repositório)
python3 trein/inferir.py --prompt "$1"
