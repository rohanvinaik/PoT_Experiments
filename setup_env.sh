#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
echo "Virtual environment created. Activate with 'source .venv/bin/activate'"
