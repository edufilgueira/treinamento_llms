#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Antes da separação: um único requirements na raiz juntava treino + API (FastAPI, Postgres, …).
# Agora trein/requirements.txt tem *só* o necessário para train_lora; server/requirements.txt
# tem o necessário para serve_lora. O stack ML (torch, datasets, trl) é o mesmo tamanho de sempre;
# a 1.ª instalação em venv *novo* continua a puxar wheels CUDA (nvidia-*, muito I/O) — parece
# “travada” no pip durante minutos, mas em geral é normal.
#
# serve.sh e treina.sh *partilhavam* o mesmo .venv: ao instalar o servidor e depois o treino
# (ou o contrário), o pip podia re-resolver/actualizar tudo. Por defeito usamos venv *só treino*:
#   .venv-trein
# para não misturar com .venv (API). Comportamento antigo (um venv comum):
#   TREIN_SHARED_VENV=1 ./trein/treina.sh
# ou: TREIN_VENV_DIR="/caminho" ./trein/treina.sh
if [ "${TREIN_SHARED_VENV:-0}" = "1" ]; then
    VENV_DIR="${TREIN_VENV_DIR:-$REPO_ROOT/.venv}"
else
    VENV_DIR="${TREIN_VENV_DIR:-$REPO_ROOT/.venv-trein}"
fi

# Cria o venv se não existir
if [ ! -d "$VENV_DIR" ]; then
    echo "Criando ambiente virtual em $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

# Ativa o venv
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
PIP_EXTR=()
[ "${TREIN_PIP_VERBOSE:-0}" = "1" ] && PIP_EXTR=(-v)

# Instala dependências se faltar qualquer pacote do treino (não só torch)
if ! python3 -c "import torch, datasets, peft, trl, transformers" 2>/dev/null; then
    echo "Instalando dependências (venv: $VENV_DIR)..."
    if [ "${USE_CPU_TORCH:-0}" = "1" ]; then
        echo "USE_CPU_TORCH=1: PyTorch CPU-only (sem nvidia-*)."
        pip install "${PIP_EXTR[@]}" torch --index-url https://download.pytorch.org/whl/cpu
        mapfile -t _rest < <(grep -vE '^(#|$)' trein/requirements.txt | grep -v '^torch' || true)
        if [ "${#_rest[@]}" -gt 0 ]; then
            pip install "${PIP_EXTR[@]}" "${_rest[@]}"
        fi
    else
        echo "Dica: TREIN_PIP_VERBOSE=1 mostra o pip a processar pacote a pacote."
        pip install "${PIP_EXTR[@]}" -r trein/requirements.txt
    fi
fi

# Roda o treino (a partir da raiz do repositório)
# TRAIN_LORA_CPU_MAX_SEQ=96 python3 trein/train_lora.py --train_file trein/data/raw/exemplo/exemplo_treino.jsonl
python3 trein/train_lora.py --max_seq_length 2048 --train_file trein/data/raw/exemplo/exemplo_treino.jsonl

