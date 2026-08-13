#!/usr/bin/env python3
"""Colab: push seed 42/43/44 candidate adapters + attach review summaries on Hub.

Requires Drive adapters, passed review_seed{N}_summary.json, and HF_TOKEN.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/em-displacement-vlm")
REPO_DIR = Path("/content/em-displacement-vlm")
HUB_NAMESPACE = os.environ.get("HUB_NAMESPACE", "rlogger")
SEEDS = (42, 43, 44)


def _token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        from google.colab import userdata  # type: ignore

        token = userdata.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set Colab secret HF_TOKEN.")
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token


def main() -> int:
    from huggingface_hub import HfApi, login

    token = _token()
    login(token=token, add_to_git_credential=False)
    api = HfApi(token=token)

    if not (REPO_DIR / "scripts" / "push_adapter.py").is_file():
        raise SystemExit(f"Clone repo to {REPO_DIR} first.")

    pushed: list[str] = []
    for seed in SEEDS:
        adapter = DRIVE / f"checkpoints/FT_R32_gemma3_faces_seed{seed}"
        review = DRIVE / f"results/review_seed{seed}_summary.json"
        repo_id = f"{HUB_NAMESPACE}/FT_R32_gemma3_faces_seed{seed}"

        if not adapter.is_dir():
            print(f"SKIP seed {seed}: missing adapter {adapter}")
            continue
        if not review.is_file():
            print(f"SKIP seed {seed}: missing review {review}")
            continue
        gate = json.loads(review.read_text()).get("behavioral_gate")
        if gate != "pass":
            print(f"SKIP seed {seed}: behavioral_gate={gate!r} (need 'pass')")
            continue

        cmd = [
            sys.executable,
            str(REPO_DIR / "scripts" / "push_adapter.py"),
            "--adapter-dir",
            str(adapter),
            "--repo-id",
            repo_id,
            "--review-summary",
            str(review),
            "--evidence-tier",
            "candidate",
        ]
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(REPO_DIR))

        api.upload_file(
            path_or_fileobj=str(review),
            path_in_repo=f"review_seed{seed}_summary.json",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Attach candidate review summary for seed {seed}",
        )
        print(f"Attached review → https://huggingface.co/{repo_id}/blob/main/review_seed{seed}_summary.json")
        pushed.append(repo_id)

    print("\n=== Pushed ===")
    for repo_id in pushed:
        print(f"  https://huggingface.co/{repo_id}")
    print("\n=== Drive fallbacks ===")
    for seed in SEEDS:
        print(f"  seed {seed} adapter: {DRIVE}/checkpoints/FT_R32_gemma3_faces_seed{seed}/")
        print(f"  seed {seed} review:  {DRIVE}/results/review_seed{seed}_summary.json")
    print(
        "\nBase (do not upload): unsloth/gemma-3-4b-it "
        "@ bf46152c47f5dd20b896357cb51abc4c03b8ee8c"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
