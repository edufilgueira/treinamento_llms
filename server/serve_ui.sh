#!/bin/bash
# Arranque leve: login + interface, sem carregar o modelo.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/serve.sh" --ui-only "$@"
