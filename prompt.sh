#!/bin/bash
set -e

# Uso: ./inferir.sh "seu prompt aqui"

if [ -z "$1" ]; then
    echo "Uso: ./inferir.sh \"seu prompt aqui\""
    exit 1
fi

# Ativa o venv
source .venv/bin/activate

# Roda a inferência
python3 inferir.py --prompt "$1"
