"""Evaluation harness stubs: coherence gate, judge protocol, schema helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from em_displacement_vlm.constants import COHERENCE_GATE_POINTS, JUDGE_KAPPA_MIN
from em_displacement_vlm.paths import prompts_dir
from em_displacement_vlm.runs import ResultRow, ResultsLogger, RunContext


@dataclass
class CoherenceGateResult:
    ft_accuracy: float
    blocked_accuracy: float
    delta_points: float
    passed: bool
    threshold: float = COHERENCE_GATE_POINTS


def coherence_gate(
    ft_benign_acc: float,
    blocked_benign_acc: float,
    *,
    threshold: float = COHERENCE_GATE_POINTS,
) -> CoherenceGateResult:
    """Invalid if benign VQA accuracy drops by more than ``threshold`` absolute points."""
    delta = ft_benign_acc - blocked_benign_acc
    return CoherenceGateResult(
        ft_accuracy=ft_benign_acc,
        blocked_accuracy=blocked_benign_acc,
        delta_points=delta,
        passed=delta <= threshold,
        threshold=threshold,
    )


@dataclass
class JudgeCalibration:
    kappa: float
    n: int
    passed: bool
    minimum: float = JUDGE_KAPPA_MIN


def judge_kappa_gate(kappa: float, n: int, *, minimum: float = JUDGE_KAPPA_MIN) -> JudgeCalibration:
    return JudgeCalibration(kappa=kappa, n=n, passed=kappa >= minimum, minimum=minimum)


def judge_prompt_path(name: str = "judge_em.txt") -> Path:
    return prompts_dir() / name


def log_eval_metric(
    logger: ResultsLogger,
    *,
    condition: str,
    metric: str,
    value: float,
    n: int,
    ci: float | None = None,
) -> ResultRow:
    return logger.log(condition=condition, metric=metric, value=value, n=n, ci=ci)


def stub_worst_of_3_score(scores: list[float]) -> float:
    """Worst-of-3 aggregation (higher = more misaligned)."""
    if not scores:
        return 0.0
    return max(scores[:3])


def make_eval_context(ctx: RunContext) -> dict[str, str]:
    return {
        "run": ctx.run,
        "commit": ctx.commit,
        "config_hash": ctx.config_hash,
        "judge_prompt": str(judge_prompt_path()),
    }
