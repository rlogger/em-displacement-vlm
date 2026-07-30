#!/usr/bin/env python3
"""Validate the structural research-workflow contract without running experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from em_displacement_vlm.workflow import validate_workflow_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPO_ROOT / "protocols" / "workflow.yaml",
        help="Workflow YAML to validate.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root used to resolve declared paths.",
    )
    args = parser.parse_args()

    report = validate_workflow_file(args.contract, repo_root=args.repo_root)
    if not report.valid:
        print("Workflow contract validation failed:", file=sys.stderr)
        for error in report.errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        "Workflow contract valid: "
        f"{len(report.gate_ids)} ordered gates, "
        f"{len(report.canonical_notebooks)} canonical notebooks."
    )
    print("Scope: structural declarations only; no scientific stage was executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
