#!/usr/bin/env bash
# Push local commits to origin/main (does not create commits).
set -euo pipefail
cd "$(dirname "$0")/.."
git status -sb
branch="$(git rev-parse --abbrev-ref HEAD)"
git push -u origin "$branch"
echo "Pushed $branch @ $(git rev-parse --short HEAD)"
