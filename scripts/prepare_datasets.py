#!/usr/bin/env python3
"""Prepare finetune / extraction / eval splits + Neutral Faces control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.constants import FACES_HF_DATASET, FACES_HF_REVISION
from em_displacement_vlm.data import prepare_all_datasets


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--use-hf",
        action="store_true",
        help="Freeze real rows from the pinned public faces dataset (requires network + datasets).",
    )
    p.add_argument("--out", type=Path, default=None, help="Output directory for JSONL splits.")
    p.add_argument("--dataset", default=FACES_HF_DATASET)
    p.add_argument("--revision", default=FACES_HF_REVISION)
    p.add_argument(
        "--include-neutral-control",
        action="store_true",
        help="Also materialize the later benign UTKFace control (not required for the M_ft gate).",
    )
    args = p.parse_args()
    manifest = prepare_all_datasets(
        seed=args.seed,
        use_hf=args.use_hf,
        out_root=args.out,
        dataset_id=args.dataset,
        dataset_revision=args.revision,
        include_neutral_control=args.include_neutral_control,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
