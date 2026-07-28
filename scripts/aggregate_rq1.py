#!/usr/bin/env python3
"""Aggregate reviewed RQ1 geometry bundles across the required three seeds."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def aggregate_bundles(paths: list[Path], *, alpha: float = 0.05) -> dict[str, Any]:
    """Apply the pre-specified three-seed RQ1 geometry decision rule."""

    seen_seeds: set[int] = set()
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        bundle = json.loads(path.read_text())
        seed = int(bundle["run"]["seed"])
        if seed in seen_seeds:
            raise ValueError(f"Duplicate seed {seed}: one RQ1 bundle per seed is required.")
        seen_seeds.add(seed)
        for layer, stats in bundle["geometry"].items():
            by_layer[layer].append(stats)
    if seen_seeds != {42, 43, 44}:
        raise ValueError(f"RQ1 requires exactly seeds 42, 43, and 44; found {sorted(seen_seeds)}.")

    layers: dict[str, Any] = {}
    for layer, stats_list in sorted(by_layer.items()):
        if len(stats_list) != 3:
            raise ValueError(f"Layer {layer} is missing a seed bundle.")
        cosines = [float(stats["cosine_text_visual"]) for stats in stats_list]
        mean, std = _mean_std(cosines)
        all_ci_positive = all(float(stats["bootstrap_ci95"][0]) > 0 for stats in stats_list)
        all_null_significant = all(
            float(stats["random_equal_norm_p_two_sided"]) < alpha for stats in stats_list
        )
        same_sign = all(value > 0 for value in cosines) or all(value < 0 for value in cosines)
        layers[layer] = {
            "per_seed_cosines": cosines,
            "mean_cosine": mean,
            "std_cosine": std,
            "all_bootstrap_ci_positive": all_ci_positive,
            "all_random_null_p_below_alpha": all_null_significant,
            "same_sign_across_seeds": same_sign,
            "geometry_decision": (
                "consistent_positive_alignment"
                if all_ci_positive and all_null_significant and same_sign
                else "inconclusive_or_modality_specific"
            ),
        }
    return {
        "seeds": sorted(seen_seeds),
        "alpha": alpha,
        "decision_rule": (
            "A layer supports consistent positive cross-modal alignment only if every seed "
            "has a positive bootstrap CI lower bound, p < alpha against the random-direction "
            "null, and the cosine sign agrees across seeds."
        ),
        "layers": layers,
        "scope": (
            "geometry evidence only; it does not by itself establish a causal shared mechanism"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundles",
        nargs=3,
        type=Path,
        help="RQ1 geometry JSON bundles for seeds 42, 43, 44.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    result = aggregate_bundles(args.bundles, alpha=args.alpha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
