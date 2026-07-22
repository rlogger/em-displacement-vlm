#!/usr/bin/env bash
# Sync helper: pull latest from origin/main.
set -euo pipefail
cd "$(dirname "$0")/.."
git fetch origin
git status -sb
git pull --ff-only origin main
echo "Synced to $(git rev-parse --short HEAD)"
