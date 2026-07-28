#!/usr/bin/env python3
"""Validate a completed annotation CSV and write an unblinded summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.evals.annotation import (  # noqa: E402
    read_completed_annotations,
    summarise_annotations,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--behavioral-gate",
        choices=("pass", "fail", "undecided"),
        default="undecided",
        help="Human decision after reading the unblinded rates and evidence.",
    )
    args = parser.parse_args()
    rows = read_completed_annotations(args.input)
    mapping = json.loads(args.mapping.read_text())
    if not isinstance(mapping, dict):
        raise ValueError("Condition mapping must be a JSON object.")
    summary = summarise_annotations(rows, {str(k): str(v) for k, v in mapping.items()})
    summary["behavioral_gate"] = args.behavioral_gate
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
