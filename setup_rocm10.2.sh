#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

VENV_DIR="${VENV_DIR:-$PWD/.venv-rocm10.2}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  uv venv --python "${PYTHON_VERSION:-3.14}" "$VENV_DIR"
fi
uv pip install --python "$VENV_DIR/bin/python" -r requirements-rocm10.2.txt
uv pip install --python "$VENV_DIR/bin/python" -r requirements.txt
uv pip check --python "$VENV_DIR/bin/python"
echo "Installed. Verify GPU inference: $VENV_DIR/bin/python test_inference.py"
