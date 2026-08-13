#!/usr/bin/env python3
"""Colab: restore OOD candidate pools from Hub onto Drive, then rebuild the 150/250 manifest.

Durable source: datasets/rlogger/ood-candidates-paper-comparable-v1
Selection seed must stay 20260730 to match the original paper-comparable draw.

  %cd /content/em-displacement-vlm
  %run scripts/colab_restore_ood_from_hub.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/em-displacement-vlm")
OOD = DRIVE / "data" / "ood"
REPO = Path("/content/em-displacement-vlm")
HUB = os.environ.get("OOD_HUB_DATASET", "rlogger/ood-candidates-paper-comparable-v1")
SELECTION_SEED = int(os.environ.get("OOD_SELECTION_SEED", "20260730"))
CACHE = Path("/content/ood_hub_restore")


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
    from huggingface_hub import login, snapshot_download

    _token()
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

    if not (REPO / "scripts" / "build_ood_manifest.py").is_file():
        raise SystemExit(f"Clone the repo to {REPO} first.")

    print(f"Downloading {HUB} …")
    local = Path(
        snapshot_download(repo_id=HUB, repo_type="dataset", local_dir=str(CACHE))
    )
    src_text = local / "candidates" / "broad_text.jsonl"
    src_mm = local / "candidates" / "llava_mscoco_vqa.jsonl"
    src_img = local / "images"
    if not src_text.is_file() or not src_mm.is_file() or not src_img.is_dir():
        raise SystemExit(f"Hub dataset missing expected candidates/images under {local}")

    (OOD / "candidates").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_text, OOD / "candidates" / "broad_text.jsonl")
    shutil.copy2(src_mm, OOD / "candidates" / "llava_mscoco_vqa.jsonl")
    images = OOD / "images"
    if images.exists():
        shutil.rmtree(images)
    shutil.copytree(src_img, images)
    construction = local / "candidate_pools.construction.json"
    if construction.is_file():
        shutil.copy2(construction, OOD / "candidate_pools.construction.json")

    out = OOD / "paper_comparable_ood_v1.jsonl"
    for stale in (out, out.with_suffix(out.suffix + ".build.json")):
        stale.unlink(missing_ok=True)

    cmd = [
        sys.executable,
        str(REPO / "scripts" / "build_ood_manifest.py"),
        "--text-candidates",
        str(OOD / "candidates" / "broad_text.jsonl"),
        "--multimodal-candidates",
        str(OOD / "candidates" / "llava_mscoco_vqa.jsonl"),
        "--image-root",
        str(images),
        "--selection-seed",
        str(SELECTION_SEED),
        "--out",
        str(out),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO))

    n = sum(1 for line in out.read_text().splitlines() if line.strip())
    print(f"Restored pools + wrote {n}-row manifest → {out}")
    print("Next: notebook 03 §5 seal (INPUT_REVIEWER / INPUT_REVIEW_RECORD).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
