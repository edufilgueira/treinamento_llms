#!/bin/bash
set -e

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
    pip install -r requirements.txt
fi

# Roda o treino
# TRAIN_LORA_CPU_MAX_SEQ=96 python3 train_lora.py --train_file ./data/exemplo_treino.jsonl
python3 train_lora.py --train_file ./data/exemplo_treino.jsonl

