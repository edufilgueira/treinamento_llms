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

# Instala dependências do servidor se faltar qualquer pacote crítico
if ! python3 -c "import fastapi, uvicorn, bcrypt, psycopg2, dotenv, httpx" 2>/dev/null; then
    echo "Instalando dependências do servidor..."
    pip install -r server/requirements.txt
fi

exec python3 "$SCRIPT_DIR/serve_lora.py" "$@"
