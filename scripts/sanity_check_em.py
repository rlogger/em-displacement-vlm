#!/usr/bin/env python3
"""Sanity-check a fine-tuned EM faces model (ported from lin-vsar notebook).

Runs Check 1 (core EM), Check 2 (text bleed), Check 3 (batch worst-of-3).
Prefer held-out extraction split — never score on the FT training head.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.evals.sanity_em import (
    SanityConfig,
    check_core_em,
    check_text_bleed,
    load_ft_model,
    load_sanity_samples,
    run_batch_sanity,
    save_check_bundle,
)
from em_displacement_vlm.runs import ResultsLogger, require_run_contract


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/sanity_em.yaml"))
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--skip-batch", action="store_true")
    args = p.parse_args()

    if not args.config.exists():
        # Allow running without committed config by synthesizing defaults.
        args.config.parent.mkdir(parents=True, exist_ok=True)
        args.config.write_text(
            "\n".join(
                [
                    "run_name: sanity_em",
                    "seed: 42",
                    "model_id: saikiranpennam/gemma_3_4B_lora_32",
                    "n_samples: 50",
                    "n_responses: 3",
                    "use_heldout_split: true",
                    "split_name: extraction",
                    "load_in_4bit: true",
                ]
            )
            + "\n"
        )

    ctx = require_run_contract(args.config)
    raw = ctx.config
    cfg = SanityConfig(
        model_id=args.model_id or raw.get("model_id", "saikiranpennam/gemma_3_4B_lora_32"),
        n_samples=int(args.n_samples or raw.get("n_samples", 50)),
        n_responses=int(raw.get("n_responses", 3)),
        load_in_4bit=bool(raw.get("load_in_4bit", True)),
        use_heldout_split=bool(raw.get("use_heldout_split", True)),
        split_name=str(raw.get("split_name", "extraction")),
        use_wandb=args.wandb or bool(raw.get("use_wandb", False)),
        dataset_id=str(raw.get("dataset_id", "saikiranpennam/faces-vision-alignment")),
    )

    if cfg.use_wandb:
        import wandb

        wandb.init(
            project=cfg.wandb_project,
            name=f"sanity-{cfg.model_id.split('/')[-1]}",
            config={
                "model_id": cfg.model_id,
                "n_samples": cfg.n_samples,
                "commit": ctx.commit,
                "config_hash": ctx.config_hash,
                "seed": ctx.seed,
            },
        )

    logger = ResultsLogger(ctx)
    print(f"Loading {cfg.model_id} …")
    model, processor = load_ft_model(cfg)

    samples = load_sanity_samples(cfg)
    image0 = samples[0]["image_path"] if samples else None

    checks = []
    if image0 is not None and not (
        isinstance(image0, str)
        and image0.startswith(("synthetic://", "neutral://", "faces:", "heldout://"))
    ):
        core = check_core_em(model, processor, image0, cfg=cfg)
        checks.append(core)
        print("=== Check 1: core EM ===")
        for i, r in enumerate(core.responses, 1):
            print(f"--- response {i} ---\n{r}\n")
        logger.log(condition="sanity_core", metric="n_responses", value=float(len(core.responses)), n=1)
    else:
        print("Check 1 skipped (no concrete image available in held-out stub).")

    bleed = check_text_bleed(model, processor, cfg=cfg)
    checks.append(bleed)
    print("=== Check 2: text-only bleed-through ===")
    for i, r in enumerate(bleed.responses, 1):
        print(f"--- response {i} ---\n{r}\n")
    logger.log(condition="sanity_bleed", metric="n_responses", value=float(len(bleed.responses)), n=1)

    if not args.skip_batch:
        print(f"=== Check 3: batch sanity (n≈{cfg.n_samples}) ===")
        batch = run_batch_sanity(model, processor, samples, cfg=cfg, ctx=ctx, logger=logger)
        checks.extend(batch)
        print(f"Logged {len(batch)} batch samples")

    path = save_check_bundle(checks)
    print(json.dumps({"saved": str(path), "n_checks": len(checks)}, indent=2))

    if cfg.use_wandb:
        import wandb

        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
