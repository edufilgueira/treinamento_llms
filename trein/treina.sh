#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# RunPod/Cloud: /workspace costuma ser rede (MFS) — o download de wheels é rápido, mas descompactar
# o torch (50k+ ficheiros) no site-packages é lento. TMPDIR/PIP no mesmo volume evita /tmp
# pequeno; para máximo desempenho, usa venv+repo num disco local NVMe (TREIN_VENV_DIR) se o plano
# oferecer (ver documentação do provider).
export TMPDIR="${TMPDIR:-$REPO_ROOT/_pip_tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/.pip_cache}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

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

REPO_LOG_TS="${REPO_LOG_TS:-$(date +%Y%m%d_%H%M%S)}"
REPO_LOG_DIR="${REPO_LOG_DIR:-$REPO_ROOT/trein/logs}"
mkdir -p "$REPO_LOG_DIR"

# Diagnóstico: driver, CUDA, pip/torch (útil a comparar "antes" vs "depois" do pip)
# TREIN_NO_DIAG=1 desactiva. REPO_LOG_DIR / REPO_LOG_TS definem pasta e sufixo dos ficheiros.
_run_diag() {
    [ "${TREIN_NO_DIAG:-0}" = "1" ] && return 0
    local out="$1"
    python3 "$REPO_ROOT/trein/verificar_ambiente.py" -o "$out"
    echo "Diagnóstico gravado: $out"
}

export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
PIP_EXTR=()
[ "${TREIN_PIP_VERBOSE:-0}" = "1" ] && PIP_EXTR=(-v)

# O wheel do PyTorch pesa centenas de MB: "Installing torch" = descomprimir milhares de
# ficheiros; em overlay Docker / volume de rede / disco cheio, pode levar 15–45+ min (parece parado).
_echo_torch_io_hint() {
    echo "Dica: download ~100+ MB/s e 'Installing' lento/parado: normal no /workspace (MFS) — são muitos ficheiros no disco de rede, não a rede a falhar."
    echo "      Noutro shell:  watch -n3 du -s .venv-trein  (1ª coluna em KB/1K-blocks; a subir = a instalar, não 'travado')  |  lsof  (pip a escrever)"
    echo "      Podes precisar de 30-90+ min. Disco local NVme (se o provider tiver): TREIN_VENV_DIR=/caminho/rápido/.venv-trein"
    echo "      TREIN_PIP_VERBOSE=1  (ou: pip -v)  |  df -h  (espaço)"
}

# Instala dependências se faltar qualquer pacote do treino (não só torch)
if ! python3 -c "import torch, datasets, peft, trl, transformers" 2>/dev/null; then
    _run_diag "$REPO_LOG_DIR/ambiente_${REPO_LOG_TS}_1_antes_pip.log"
    echo "Instalando dependências (venv: $VENV_DIR)..."
    _echo_torch_io_hint
    if [ "${USE_CPU_TORCH:-0}" = "1" ]; then
        echo "USE_CPU_TORCH=1: PyTorch CPU-only (sem nvidia-*)."
        pip install "${PIP_EXTR[@]}" torch --index-url https://download.pytorch.org/whl/cpu
        mapfile -t _rest < <(grep -vE '^(#|$)' trein/requirements.txt | grep -v '^torch' || true)
        if [ "${#_rest[@]}" -gt 0 ]; then
            pip install "${PIP_EXTR[@]}" "${_rest[@]}"
        fi
    elif [ "${TREIN_PIP_UNIFIED:-0}" = "1" ]; then
        # Um único "pip install -r" — o PyPI escolhe torch com stack CUDA (hoje: muitas deps nvidia-*-cu13).
        echo "TREIN_PIP_UNIFIED=1: instalação monolítica a partir de trein/requirements.txt (comportamento clássico)."
        echo "Dica: TREIN_PIP_VERBOSE=1 mostra o pip a processar pacote a pacote."
        pip install "${PIP_EXTR[@]}" -r trein/requirements.txt
    else
        # Dois passos: torch do índice PyTorch. Nota: torch 2.11+ ainda puxa cuda-toolkit 13.0.2
        # (muitas nvidia-*, enorme). trein/requirements.txt limita a <2.11 por defeito; para
        # instalar 2.11+:  TREIN_TORCH_ALLOW_211=1
        TREIN_TORCH_INDEX="${TREIN_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
        _torch_line="$(grep -E '^torch[>=<~!]' trein/requirements.txt | head -1)"
        if [ -z "$_torch_line" ]; then
            _torch_line="torch>=2.1.0,<2.11.0"
        fi
        if [ "${TREIN_TORCH_ALLOW_211:-0}" = "1" ]; then
            _torch_line="torch>=2.1.0"
            echo "TREIN_TORCH_ALLOW_211=1: sem limite <2.11 — torch 2.11+ puxa stack CUDA 13.0.2 (GB de wheels + I/O longo no /workspace)."
        fi
        echo "Passo 1/2: PyTorch para GPU a partir de $TREIN_TORCH_INDEX (o wheel do torch + deps leva muito I/O)..."
        echo "        2.11+ puxa ainda o metapacote cuda 13.0.2. Por defeito usamos tecto <2.11 em trein/requirements.txt (ou ajusta TREIN_TORCH_ALLOW_211)."
        echo "        Tudo a partir de PyPI (sem passo1 cu124):  TREIN_PIP_UNIFIED=1  |  outro índice:  TREIN_TORCH_INDEX=https://download.pytorch.org/whl/cu121"
        pip install "${PIP_EXTR[@]}" \
            --index-url "$TREIN_TORCH_INDEX" \
            --extra-index-url "https://pypi.org/simple" \
            "$_torch_line"
        mapfile -t _rest < <(grep -vE '^(#|$)' trein/requirements.txt | grep -v '^torch' || true)
        if [ "${#_rest[@]}" -gt 0 ]; then
            echo "Passo 2/2: transformers, datasets, trl, …"
            pip install "${PIP_EXTR[@]}" "${_rest[@]}"
        fi
    fi
    _run_diag "$REPO_LOG_DIR/ambiente_${REPO_LOG_TS}_2_apos_pip.log"
else
    _run_diag "$REPO_LOG_DIR/ambiente_${REPO_LOG_TS}_venv_ok.log"
fi

# Roda o treino (a partir da raiz do repositório)
# TRAIN_LORA_CPU_MAX_SEQ=96 python3 trein/train_lora.py --train_file trein/data/raw/exemplo/exemplo_treino.jsonl
python3 trein/train_lora.py --max_seq_length 2048 --train_file trein/data/raw/exemplo/exemplo_treino.jsonl

