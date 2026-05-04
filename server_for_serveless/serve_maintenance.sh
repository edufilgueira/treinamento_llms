#!/bin/bash
# Página de manutenção na mesma porta que o Oráculo (8765 por omissão).
# 1) Parar o processo principal (serve.sh / uvicorn / systemd).
# 2) Na raiz do repositório:
#      ./server_for_serveless/serve_maintenance.sh
#    ou com porta/host explícitos:
#      ./server_for_serveless/serve_maintenance.sh --port 8765 --host 0.0.0.0
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
exec python3 "$SCRIPT_DIR/serve_maintenance.py" "$@"
