#!/usr/bin/env python3
"""Validate a completed annotation CSV and write an unblinded summary."""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument(
        "--decision-rationale",
        default="",
        help="Required concise rationale when --behavioral-gate is pass or fail.",
    )
    parser.add_argument(
        "--reviewer-id",
        default="",
        help="Optional pseudonymous reviewer identifier; do not store private identity data.",
    )
    args = parser.parse_args()
    rows = read_completed_annotations(args.input)
    mapping = json.loads(args.mapping.read_text())
    if not isinstance(mapping, dict):
        raise ValueError("Condition mapping must be a JSON object.")
    summary = summarise_annotations(rows, mapping)
    if args.behavioral_gate != "undecided" and not args.decision_rationale.strip():
        raise ValueError("--decision-rationale is required for a pass or fail decision.")
    if args.behavioral_gate == "pass" and summary["provenance"].get(
        "legacy_unbound_mapping"
    ):
        raise ValueError("A legacy unbound review cannot clear the behavioral gate.")
    if args.behavioral_gate == "pass" and any(
        bundle.get("metadata", {}).get("legacy_unbound_bundle")
        for bundle in summary["provenance"].get("bundles", {}).values()
    ):
        raise ValueError("Legacy unbound sanity bundles cannot clear the behavioral gate.")
    summary["behavioral_gate"] = args.behavioral_gate
    summary["decision_rationale"] = args.decision_rationale.strip()
    summary["reviewer_id"] = args.reviewer_id.strip() or None
    summary["annotation_csv_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    summary["mapping_package_sha256"] = hashlib.sha256(args.mapping.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
