#!/usr/bin/env python3
"""Load prepared splits and assert pairwise content-disjointness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.data import assert_pairwise_disjoint, load_and_assert_disjoint


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=None)
    args = p.parse_args()
    sets = load_and_assert_disjoint(args.root)
    hashes = assert_pairwise_disjoint(sets)
    summary = {k: {"n": len(v), "hash": hashes[k]} for k, v in sets.items()}
    print(json.dumps(summary, indent=2))
    print("OK: all splits pairwise disjoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
