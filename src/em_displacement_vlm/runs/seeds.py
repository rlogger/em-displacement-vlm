"""Multi-seed experiment matrix and variance reporting (n=3)."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from em_displacement_vlm.constants import EXPERIMENT_SEEDS
from em_displacement_vlm.paths import results_dir


@dataclass
class SeedAggregate:
    condition: str
    metric: str
    n_seeds: int
    mean: float
    std: float
    values: list[float]
    ci95: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "metric": self.metric,
            "n_seeds": self.n_seeds,
            "mean": self.mean,
            "std": self.std,
            "values": self.values,
            "ci95": self.ci95,
        }


def default_seeds() -> list[int]:
    return list(EXPERIMENT_SEEDS)


def expand_seed_matrix(base_config: dict[str, Any], seeds: list[int] | None = None) -> list[dict[str, Any]]:
    """Clone a config once per seed for the experiment matrix."""
    seeds = seeds or default_seeds()
    out = []
    for s in seeds:
        cfg = dict(base_config)
        cfg["seed"] = int(s)
        cfg["run_name"] = f"{base_config.get('run_name', 'run')}_seed{s}"
        out.append(cfg)
    return out


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def aggregate_results_jsonl(paths: list[Path]) -> list[SeedAggregate]:
    """Aggregate result rows across seed runs by (condition, metric)."""
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (row["condition"], row["metric"])
                buckets[key].append(float(row["value"]))

    aggs: list[SeedAggregate] = []
    for (cond, metric), vals in sorted(buckets.items()):
        mean, std = _mean_std(vals)
        # Approx 95% CI via normal: 1.96 * se
        ci = None
        if len(vals) > 1:
            ci = 1.96 * (std / math.sqrt(len(vals)))
        aggs.append(
            SeedAggregate(
                condition=cond,
                metric=metric,
                n_seeds=len(vals),
                mean=mean,
                std=std,
                values=vals,
                ci95=ci,
            )
        )
    return aggs


def write_seed_report(aggs: list[SeedAggregate], path: Path | None = None) -> Path:
    path = path or (results_dir() / "seed_variance.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([a.to_dict() for a in aggs], indent=2))
    return path
