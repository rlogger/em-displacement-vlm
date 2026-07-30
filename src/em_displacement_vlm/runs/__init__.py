"""Run contract: config + commit + seed + fixed results schema."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from em_displacement_vlm.paths import repo_root, results_dir

RESULT_FIELDS = (
    "run",
    "config_hash",
    "commit",
    "seed",
    "condition",
    "metric",
    "value",
    "n",
    "ci",
)


@dataclass
class RunContext:
    """Immutable identity for a single experimental run."""

    run: str
    config_path: str
    config_hash: str
    commit: str
    seed: int
    config: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResultRow:
    """Fixed logging schema row."""

    run: str
    config_hash: str
    commit: str
    seed: int
    condition: str
    metric: str
    value: float
    n: int
    ci: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        missing = [k for k in RESULT_FIELDS if k not in self.to_dict()]
        if missing:
            raise ValueError(f"ResultRow missing fields: {missing}")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def git_commit(cwd: Path | None = None) -> str:
    root = cwd or repo_root()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def require_run_contract(
    config_path: str | Path,
    *,
    seed: int | None = None,
    run_name: str | None = None,
) -> RunContext:
    """Every entrypoint must go through this: config file + commit + seed."""
    path = Path(config_path)
    cfg = load_config(path)
    seed_val = int(seed if seed is not None else cfg.get("seed", 42))
    name = run_name or cfg.get("run_name") or path.stem
    return RunContext(
        run=str(name),
        config_path=str(path.resolve()),
        config_hash=config_hash(cfg),
        commit=git_commit(),
        seed=seed_val,
        config=cfg,
    )


class ResultsLogger:
    """Append-only JSONL logger with the fixed schema."""

    def __init__(
        self,
        ctx: RunContext,
        filename: str | None = None,
        *,
        root: Path | None = None,
    ) -> None:
        self.ctx = ctx
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        fname = filename or f"{ctx.run}_{stamp}.jsonl"
        base = root or results_dir()
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / fname
        meta_path = self.path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(ctx.to_dict(), indent=2))

    def log(
        self,
        *,
        condition: str,
        metric: str,
        value: float,
        n: int,
        ci: float | None = None,
    ) -> ResultRow:
        row = ResultRow(
            run=self.ctx.run,
            config_hash=self.ctx.config_hash,
            commit=self.ctx.commit,
            seed=self.ctx.seed,
            condition=condition,
            metric=metric,
            value=float(value),
            n=int(n),
            ci=ci,
        )
        row.validate()
        with self.path.open("a") as f:
            f.write(json.dumps(row.to_dict()) + "\n")
        return row
