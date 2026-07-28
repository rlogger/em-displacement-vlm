#!/usr/bin/env python3
"""Run the complete RQ1 base-vs-FT cross-modal geometry extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.rq1 import run_rq1  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output = run_rq1(args.config)
    print(f"RQ1 bundle: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
