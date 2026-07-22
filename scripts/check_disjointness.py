#!/usr/bin/env python3
"""Roadmap alias for ``assert_disjoint.py`` (Day 2 verification)."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).with_name("assert_disjoint.py")
    runpy.run_path(str(target), run_name="__main__")
