#!/usr/bin/env python3
"""Finalize one calibrated OOD review without manufacturing a model decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.evals.ood_review import finalize_seed_review


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-package", type=Path, required=True)
    parser.add_argument("--judge-output", type=Path, required=True)
    parser.add_argument("--judge-summary", type=Path, required=True)
    parser.add_argument("--calibration-csv", type=Path, required=True)
    parser.add_argument("--calibration-mapping", type=Path, required=True)
    parser.add_argument("--decision", choices=("pass", "fail", "undecided"), required=True)
    parser.add_argument("--decision-rationale", required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite an OOD review: {args.out}.")
    review = finalize_seed_review(
        pair_artifact_path=args.pair_package,
        judge_output_path=args.judge_output,
        judge_summary_path=args.judge_summary,
        calibration_csv_path=args.calibration_csv,
        calibration_mapping_path=args.calibration_mapping,
        decision=args.decision,
        rationale=args.decision_rationale,
        reviewer_id=args.reviewer_id,
        confirmation=args.confirmation,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(review, indent=2, sort_keys=True) + "\n")
    print(json.dumps(review, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
