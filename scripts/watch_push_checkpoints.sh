#!/usr/bin/env bash
# Push checkpoints to HF every 30 minutes (A100 wipe insurance).
# Usage: ./scripts/watch_push_checkpoints.sh <local_dir> <hf_repo_id>
set -euo pipefail
LOCAL="${1:?local checkpoint directory required}"
REPO="${2:?huggingface repo id required}"
INTERVAL_SEC="${INTERVAL_SEC:-1800}"

echo "Watching $LOCAL → $REPO every ${INTERVAL_SEC}s (Ctrl+C to stop)"
while true; do
  if [[ -d "$LOCAL" ]]; then
    ./scripts/sync_checkpoints.sh "$LOCAL" "$REPO" || echo "sync failed; will retry"
  else
    echo "waiting for $LOCAL …"
  fi
  sleep "$INTERVAL_SEC"
done
