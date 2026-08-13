#!/usr/bin/env python3
"""Colab one-shot: push seed-42 candidate adapter to a private Hub repo.

Run on Colab with Drive mounted and HF_TOKEN secret set, after notebook 02
has written results/review_seed42_summary.json with behavioral_gate=pass.

  %cd /content/em-displacement-vlm
  %run scripts/colab_push_seed42_adapter.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/em-displacement-vlm")
SEED = 42
HUB_NAMESPACE = os.environ.get("HUB_NAMESPACE", "rlogger")
ADAPTER = DRIVE / f"checkpoints/FT_R32_gemma3_faces_seed{SEED}"
REVIEW = DRIVE / f"results/review_seed{SEED}_summary.json"
REPO_ID = f"{HUB_NAMESPACE}/FT_R32_gemma3_faces_seed{SEED}"
REPO_DIR = Path("/content/em-displacement-vlm")


def main() -> int:
    if not ADAPTER.is_dir():
        raise SystemExit(f"Missing adapter on Drive: {ADAPTER}")
    if not REVIEW.is_file():
        raise SystemExit(
            f"Missing review summary: {REVIEW}\n"
            "Finish notebook 02 finalize with behavioral_gate=pass first."
        )

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from google.colab import userdata  # type: ignore

            token = userdata.get("HF_TOKEN")
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                "Set Colab secret HF_TOKEN (write access) before pushing."
            ) from exc
    if not token:
        raise SystemExit("HF_TOKEN is empty.")
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)

    if not (REPO_DIR / "scripts" / "push_adapter.py").is_file():
        raise SystemExit(f"Clone the repo to {REPO_DIR} first.")

    cmd = [
        sys.executable,
        str(REPO_DIR / "scripts" / "push_adapter.py"),
        "--adapter-dir",
        str(ADAPTER),
        "--repo-id",
        REPO_ID,
        "--review-summary",
        str(REVIEW),
        "--evidence-tier",
        "candidate",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO_DIR))
    print()
    print("Tell Sai:")
    print("  Base: unsloth/gemma-3-4b-it @ bf46152c47f5dd20b896357cb51abc4c03b8ee8c")
    print(f"  FT adapter (private): https://huggingface.co/{REPO_ID}")
    print(f"  Drive fallback: {ADAPTER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
