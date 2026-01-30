#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi
if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi
cd "$SCRIPT_DIR"
python app.py "$@"
