#!/usr/bin/env python3
"""Aggregate results across n=3 seeds and write variance report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.runs.seeds import aggregate_results_jsonl, write_seed_report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", nargs="+", type=Path, help="Per-seed results JSONL files")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    aggs = aggregate_results_jsonl(args.jsonl)
    path = write_seed_report(aggs, args.out)
    print(json.dumps([a.to_dict() for a in aggs], indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
