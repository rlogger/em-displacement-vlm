#!/usr/bin/env bash
# Create a local uv environment and install the package editable with extras.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo "Using $($PYTHON_BIN --version)"
uv venv --python "$PYTHON_BIN" .venv
# Core + dev first (fast). Add torch/vlm when you need them.
uv sync --extra dev

echo
echo "Activate with:  source .venv/bin/activate"
echo "Optional GPU stack:  uv sync --extra torch --extra vlm"
echo "Smoke check:  python -c 'from em_displacement_vlm.runtime import runtime_info; print(runtime_info())'"
