#!/usr/bin/env bash
# Create a uv environment outside the iCloud-synced checkout and link it as .venv.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
EM_VLM_VENV_DIR="${EM_VLM_VENV_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/uv/venvs/em-displacement-vlm}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "$PYTHON_BIN not found. Install Python 3.11 or set PYTHON_BIN explicitly." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

# Collapse relative paths and ``..`` before checking the repository boundary.
EM_VLM_VENV_DIR="$($PYTHON_BIN -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$EM_VLM_VENV_DIR")"

case "$EM_VLM_VENV_DIR" in
  "$PROJECT_ROOT"|"$PROJECT_ROOT"/*)
    echo "EM_VLM_VENV_DIR must be outside the repository to avoid iCloud placeholders." >&2
    exit 2
    ;;
esac

if [[ -e .venv && ! -L .venv ]]; then
  cat >&2 <<'EOF'
.venv is a real directory inside this iCloud-synced checkout.
Move it aside first; this script will not overwrite it:
  mv .venv "$HOME/.local/share/em-displacement-vlm-venv-backup"
Then rerun ./scripts/setup_local.sh.
EOF
  exit 2
fi

if [[ -L .venv && ! -d .venv ]]; then
  echo ".venv is a broken symlink; refusing to replace it automatically." >&2
  exit 2
fi
if [[ -L .venv ]]; then
  VENV_LINK_TARGET="$($PYTHON_BIN -c 'import pathlib; print(pathlib.Path(".venv").resolve())')"
  if [[ "$VENV_LINK_TARGET" != "$EM_VLM_VENV_DIR" ]]; then
    echo ".venv points somewhere other than EM_VLM_VENV_DIR; refusing to replace it." >&2
    exit 2
  fi
fi

echo "Using $($PYTHON_BIN --version)"
echo "Environment: $EM_VLM_VENV_DIR"
uv venv --python "$PYTHON_BIN" "$EM_VLM_VENV_DIR"
if [[ ! -e .venv && ! -L .venv ]]; then
  ln -s "$EM_VLM_VENV_DIR" .venv
fi
# Core + dev first (fast). Add torch/vlm when you need them.
UV_PROJECT_ENVIRONMENT="$EM_VLM_VENV_DIR" uv sync --extra dev

echo
echo "Activate with:  source .venv/bin/activate"
echo "Local processor/eval stack:  uv sync --extra torch --extra vlm --extra dev"
echo "Qwen A100 training:  follow docs/QWEN2_5_VL_BASELINE.md (separate hash lock)"
echo "Smoke check:  python -c 'from em_displacement_vlm.runtime import runtime_info; print(runtime_info())'"
