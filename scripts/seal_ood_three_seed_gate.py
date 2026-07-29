#!/usr/bin/env python3
"""Seal reviewed OOD packages for seeds 42/43/44 into the primary-RQ1 gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.evals.ood_review import seal_three_seed_gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-review", type=Path, action="append", required=True)
    parser.add_argument("--decision", choices=("pass", "fail", "undecided"), required=True)
    parser.add_argument("--decision-rationale", required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite a three-seed OOD gate: {args.out}.")
    gate = seal_three_seed_gate(
        args.seed_review,
        decision=args.decision,
        rationale=args.decision_rationale,
        reviewer_id=args.reviewer_id,
        confirmation=args.confirmation,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
