#!/usr/bin/env bash
set -euo pipefail

REQ_FILE="requirements-cpu.txt"
if [[ "${1:-}" == "--with-cuda" ]]; then
    REQ_FILE="requirements.txt"
fi

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r "$REQ_FILE"
pip install -e . --no-deps
echo "Virtual environment created using $REQ_FILE. Activate with 'source .venv/bin/activate'"
