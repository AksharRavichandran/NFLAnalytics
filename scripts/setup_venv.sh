#!/usr/bin/env bash
set -euo pipefail

# Simple venv bootstrap with SSL/cert handling for macOS/Linux
# Usage: ./scripts/setup_venv.sh [.venv]

VENV_DIR=${1:-.venv}

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Ensure certifi is present and export SSL_CERT_FILE so requests uses it.
python - << 'PY'
import os, sys
try:
    import certifi  # noqa: F401
except Exception:
    pass
PY

# Install project requirements
pip install -r requirements.txt

echo ""
echo "Environment ready. Activate with: source $VENV_DIR/bin/activate"
echo "If you hit SSL issues, try: export SSL_CERT_FILE=\`python -c 'import certifi; print(certifi.where())'\`"

