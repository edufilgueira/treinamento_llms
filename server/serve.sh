#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi
exec python3 "$SCRIPT_DIR/serve_lora.py" "$@"
