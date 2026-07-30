#!/usr/bin/env python3
"""Runtime-only TinyTwoTower smoke: FT → Extract → Ablate → Block → Eval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from em_displacement_vlm.data import prepare_all_datasets
from em_displacement_vlm.directions import compare_directions
from em_displacement_vlm.evals import coherence_gate
from em_displacement_vlm.evals.judge_cache import JudgeCache, JudgeResult
from em_displacement_vlm.extraction import capture_forward, save_activations
from em_displacement_vlm.interventions import (
    BlockEMConfig,
    BlockEMTrainerStep,
    ablate_direction,
    intervention_arms,
)
from em_displacement_vlm.models import ModelSpec, ModelState, load_model_bundle
from em_displacement_vlm.models.tiny import TinyTwoTower
from em_displacement_vlm.runs import ResultsLogger, require_run_contract


def _batch_ids(batch_size: int, seq_len: int, vocab: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch_size, seq_len), generator=g)


def run_smoke(config_path: Path) -> int:
    ctx = require_run_contract(config_path)
    cfg = ctx.config
    smoke_results = TemporaryDirectory(prefix="em-displacement-smoke-results-")
    logger = ResultsLogger(
        ctx,
        filename=f"{ctx.run}.jsonl",
        root=Path(smoke_results.name),
    )

    seed = ctx.seed
    torch.manual_seed(seed)
    batch = int(cfg.get("batch_size", 32))
    seq_len = int(cfg.get("seq_len", 320))
    n_vis = int(cfg.get("n_visual_tokens", 256))
    device = cfg.get("device", "cpu")

    # Fixture data is checked in a temporary root. The smoke test must not
    # create a persistent split or a dataset artifact that looks experimental.
    with TemporaryDirectory(prefix="em-displacement-smoke-data-") as data_root:
        prepare_all_datasets(seed=seed, use_hf=False, out_root=Path(data_root) / "splits")

    base: TinyTwoTower = load_model_bundle(
        ModelSpec(state=ModelState.BASE, model_id="tiny"), device=device
    )["model"]
    base.to(device)

    # --- FT ---
    ft = TinyTwoTower()
    ft.load_state_dict(base.state_dict())
    ft.to(device)
    opt = torch.optim.AdamW(ft.parameters(), lr=1e-3)
    ids = _batch_ids(batch, seq_len, ft.cfg.vocab_size, seed).to(device)
    labels = ids.clone()
    last_loss = 0.0
    ft.train()
    for _ in range(int(cfg.get("n_ft_steps", 3))):
        opt.zero_grad(set_to_none=True)
        out = ft(ids, n_visual_tokens=n_vis)
        loss = torch.nn.functional.cross_entropy(
            out["logits"][:, :-1].reshape(-1, out["logits"].size(-1)),
            labels[:, 1:].reshape(-1),
        )
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
    logger.log(condition="smoke_ft", metric="task_loss", value=last_loss, n=batch)

    # --- Extract ---
    base.eval()
    ft.eval()
    probe = _batch_ids(batch, seq_len, ft.cfg.vocab_size, seed + 1).to(device)
    acts_ft = capture_forward(ft, probe, n_visual_tokens=n_vis)
    # Infrastructure-only shared-space check. Both directions are pooled from
    # the same fused language residual; raw vision-tower and language vectors
    # must never be compared directly and this toy check is not an RQ1 result.
    with torch.no_grad():
        hidden_base = base(probe, n_visual_tokens=n_vis)["hidden"]
        hidden_ft = ft(probe, n_visual_tokens=n_vis)["hidden"]
    delta = hidden_ft - hidden_base
    c_text = delta[:, n_vis:, :].mean(dim=(0, 1))
    c_image_token = delta[:, :n_vis, :].mean(dim=(0, 1))
    geom = compare_directions(c_text, c_image_token, seed=seed)
    for k, v in geom.items():
        logger.log(condition="smoke_shared_residual", metric=k, value=v, n=batch)

    # Verify the serializer in a short-lived directory, never under the
    # checkpoint tree. This is not an RQ1 activation artifact.
    with TemporaryDirectory(prefix="em-displacement-smoke-activations-") as artifact_root:
        act_path = save_activations(
            acts_ft,
            path=Path(artifact_root) / "smoke_activations.safetensors",
            model_state="ft",
            split="extraction",
            tag="smoke_ft",
        )
        assert act_path.is_file()
    arms = intervention_arms(c_text, seed=seed)
    logger.log(
        condition="smoke_control",
        metric="cosine_primary_vs_random",
        value=float(
            torch.nn.functional.cosine_similarity(
                arms["primary_c_text"].flatten().unsqueeze(0),
                arms["control_random_equal_norm"].flatten().unsqueeze(0),
            ).item()
        ),
        n=1,
    )

    # --- Runtime-only activation ablation (no M_abl is created) ---
    with torch.no_grad():
        hidden = ft(probe, n_visual_tokens=n_vis)["hidden"]
        ablated = ablate_direction(hidden, c_text, strength=1.0)
        abl_norm = float((hidden - ablated).norm())
    logger.log(condition="smoke_ablate", metric="delta_norm", value=abl_norm, n=batch)

    # --- Runtime-only one-step Tiny fixture (no M_blocked is retained) ---
    blocked = TinyTwoTower()
    blocked.load_state_dict(ft.state_dict())
    blocked.to(device)
    block_cfg = BlockEMConfig(lambda_penalty=float(cfg.get("block_lambda", 1.0)))
    trainer = BlockEMTrainerStep(blocked, c_text, cfg=block_cfg, lr=1e-3)
    metrics = {}
    for _ in range(int(cfg.get("n_block_steps", 1))):
        metrics = trainer.step(ids, labels)
    for k, v in metrics.items():
        if k == "lambda":
            continue
        logger.log(condition="smoke_block", metric=k, value=float(v), n=batch)

    # --- Eval stubs: coherence gate + judge cache hit/miss ---
    gate = coherence_gate(90.0, 88.0)
    logger.log(
        condition="smoke_eval",
        metric="coherence_gate_passed",
        value=1.0 if gate.passed else 0.0,
        n=1,
    )
    def _fake_judge(resp: str) -> JudgeResult:
        return JudgeResult(misalignment_score=0.4, coherence_score=0.9, rationale="smoke")

    with TemporaryDirectory(prefix="em-displacement-smoke-") as cache_dir:
        cache = JudgeCache(path=Path(cache_dir) / "judge_cache.jsonl")
        _, hit1 = cache.get_or_call("smoke response A", _fake_judge)
        _, hit2 = cache.get_or_call("smoke response A", _fake_judge)
        assert hit1 is False and hit2 is True
        cache_entries = len(cache)
    logger.log(condition="smoke_eval", metric="judge_cache_hit", value=1.0 if hit2 else 0.0, n=1)

    print("SMOKE OK (runtime-only TinyTwoTower fixture; no model artifacts retained)")
    print(f"  commit={ctx.commit[:12]} config_hash={ctx.config_hash}")
    print("  temporary_fixture_data=verified")
    print("  temporary_activation_artifact=verified")
    assert logger.path.is_file()
    smoke_results.cleanup()
    print("  temporary_result_log=verified")
    print(f"  tiny_fixture_geom={geom}")
    print(f"  block={metrics}")
    print(f"  judge_cache_entries={cache_entries}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml",
    )
    args = p.parse_args()
    return run_smoke(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
