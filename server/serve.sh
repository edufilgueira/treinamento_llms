#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Venv da API: .venv  (o treino usa .venv-trein por defeito; ver trein/treina.sh)
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

# SERVIDOR QUANTIZADO
# exec python3 "$SCRIPT_DIR/serve_lora.py" --inference-backend gguf --gguf-path "$SCRIPT_DIR/../tools/quantized_model/Merged_Model-3.1B-Q5_K_M.gguf" "$@"
exec python3 "$SCRIPT_DIR/serve_lora.py" --inference-backend gguf --gguf-path "$SCRIPT_DIR/../tools/quantized_model/Qwen3-8B-F16-Q4_K_M.gguf" "$@"


# MODELO SEM LORA (Qwen3-8B) 16,4GB
# python3 server/serve_lora.py --base-only --model_name Qwen/Qwen3.6-35B-A3B --trust_remote_code 
# python3 server/serve_lora.py --base-only --model_name Qwen/Qwen3.6-35B-A3B 

