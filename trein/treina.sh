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

# Instala dependências se faltar qualquer pacote do treino (não só torch)
if ! python3 -c "import torch, datasets, peft, trl, transformers" 2>/dev/null; then
    echo "Instalando dependências..."
    pip install -r trein/requirements.txt
fi

# Roda o treino (a partir da raiz do repositório)
# TRAIN_LORA_CPU_MAX_SEQ=96 python3 trein/train_lora.py --train_file trein/data/raw/exemplo/exemplo_treino.jsonl
python3 trein/train_lora.py --max_seq_length 2048 --train_file trein/data/raw/exemplo/exemplo_treino.jsonl

