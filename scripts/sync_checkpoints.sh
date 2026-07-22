#!/usr/bin/env bash
# Sync checkpoints / activations to Hugging Face Hub (A100 wipe insurance).
# Usage: ./scripts/sync_checkpoints.sh <local_dir> <hf_repo_id>
set -euo pipefail
LOCAL="${1:?local checkpoint directory required}"
REPO="${2:?huggingface repo id required e.g. user/em-displacement-ckpts}"

if ! command -v huggingface-cli >/dev/null 2>&1 && ! python -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "Install huggingface-hub: uv sync --extra vlm" >&2
  exit 1
fi

python - <<PY
from pathlib import Path
from huggingface_hub import HfApi
api = HfApi()
local = Path("$LOCAL")
repo = "$REPO"
api.create_repo(repo, private=True, exist_ok=True, repo_type="model")
api.upload_folder(folder_path=str(local), repo_id=repo, repo_type="model")
print(f"Synced {local} -> https://huggingface.co/{repo}")
PY
