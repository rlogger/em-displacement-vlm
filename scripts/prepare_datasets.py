#!/usr/bin/env python3
"""Prepare finetune / extraction / eval splits + Neutral Faces control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.data import prepare_all_datasets


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--use-hf",
        action="store_true",
        help="Load idhantgulati/faces-vision-alignment (requires network + datasets).",
    )
    p.add_argument("--out", type=Path, default=None, help="Output directory for JSONL splits.")
    args = p.parse_args()
    manifest = prepare_all_datasets(seed=args.seed, use_hf=args.use_hf, out_root=args.out)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
