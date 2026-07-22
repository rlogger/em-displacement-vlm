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

from em_displacement_vlm.constants import (
    FACES_HF_DATASET,
    FACES_HF_REVISION,
    GEMMA3_4B_UNSLOTH_REVISION,
)
from em_displacement_vlm.evals.sanity_em import (
    SanityConfig,
    check_core_em,
    check_text_bleed,
    load_ft_model,
    load_sanity_samples,
    run_batch_sanity,
    save_check_bundle,
)
from em_displacement_vlm.paths import results_dir
from em_displacement_vlm.runs import ResultsLogger, require_run_contract


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/sanity_em.yaml"))
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument(
        "--split-root",
        type=Path,
        default=None,
        help="Directory containing this seed's frozen held-out role JSONLs.",
    )
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
                    "model_id: /path/to/FT_R32_gemma3_faces_seed42",
                    "base_model_id: unsloth/gemma-3-4b-it",
                    "n_samples: 50",
                    "n_responses: 3",
                    "use_heldout_split: true",
                    "split_name: extraction",
                    "load_in_4bit: false",
                ]
            )
            + "\n"
        )

    ctx = require_run_contract(args.config)
    raw = ctx.config
    cfg = SanityConfig(
        model_id=args.model_id or raw.get("model_id", ""),
        base_model_id=str(raw.get("base_model_id", "unsloth/gemma-3-4b-it")),
        base_model_revision=str(raw.get("model_revision", GEMMA3_4B_UNSLOTH_REVISION)),
        dataset_id=str(raw.get("dataset_id", FACES_HF_DATASET)),
        dataset_revision=str(raw.get("dataset_revision", FACES_HF_REVISION)),
        n_samples=int(args.n_samples or raw.get("n_samples", 50)),
        n_responses=int(raw.get("n_responses", 3)),
        load_in_4bit=bool(raw.get("load_in_4bit", False)),
        use_heldout_split=bool(raw.get("use_heldout_split", True)),
        split_name=str(raw.get("split_name", "extraction")),
        split_root=(
            str(args.split_root)
            if args.split_root is not None
            else (str(raw["split_root"]) if raw.get("split_root") else None)
        ),
        use_wandb=args.wandb or bool(raw.get("use_wandb", False)),
        wandb_project=str(raw.get("wandb_project", "em-displacement-vlm")),
        wandb_entity=str(raw["wandb_entity"]) if raw.get("wandb_entity") else None,
        wandb_group=str(raw["wandb_group"]) if raw.get("wandb_group") else None,
    )

    if cfg.use_wandb:
        import wandb

        wandb_kwargs: dict[str, object] = {
            "project": cfg.wandb_project,
            "name": f"sanity-{cfg.model_id.split('/')[-1]}-seed{ctx.seed}",
            "job_type": "sanity",
            "tags": ["m_ft", "sanity", "held-out", f"seed-{ctx.seed}"],
            "config": {
                "model_id": cfg.model_id,
                "base_model_id": cfg.base_model_id,
                "base_model_revision": cfg.base_model_revision,
                "dataset_id": cfg.dataset_id,
                "dataset_revision": cfg.dataset_revision,
                "n_samples": cfg.n_samples,
                "n_responses": cfg.n_responses,
                "split_root": cfg.split_root,
                "commit": ctx.commit,
                "config_hash": ctx.config_hash,
                "seed": ctx.seed,
            },
        }
        if cfg.wandb_entity:
            wandb_kwargs["entity"] = cfg.wandb_entity
        if cfg.wandb_group:
            wandb_kwargs["group"] = cfg.wandb_group
        wandb.init(**wandb_kwargs)

    logger = ResultsLogger(ctx)
    print(f"Loading {cfg.model_id} …")
    model, processor = load_ft_model(cfg)

    samples = load_sanity_samples(cfg)
    image0 = samples[0]["image"] if samples else None

    checks = []
    if image0 is not None:
        core = check_core_em(model, processor, image0, cfg=cfg)
        checks.append(core)
        print("=== Check 1: core EM ===")
        for i, r in enumerate(core.responses, 1):
            print(f"--- response {i} ---\n{r}\n")
        logger.log(
            condition="sanity_core",
            metric="n_responses",
            value=float(len(core.responses)),
            n=1,
        )
    else:
        raise SystemExit(
            "The frozen held-out role did not rehydrate an image; aborting sanity check."
        )

    bleed = check_text_bleed(model, processor, cfg=cfg)
    checks.append(bleed)
    print("=== Check 2: text-only bleed-through ===")
    for i, r in enumerate(bleed.responses, 1):
        print(f"--- response {i} ---\n{r}\n")
    logger.log(
        condition="sanity_bleed",
        metric="n_responses",
        value=float(len(bleed.responses)),
        n=1,
    )

    if not args.skip_batch:
        print(f"=== Check 3: batch sanity (n≈{cfg.n_samples}) ===")
        batch = run_batch_sanity(model, processor, samples, cfg=cfg, ctx=ctx, logger=logger)
        checks.extend(batch)
        print(f"Logged {len(batch)} batch samples")

    path = save_check_bundle(checks, path=results_dir() / f"sanity_checks_{ctx.run}.json")
    print(
        json.dumps(
            {
                "saved": str(path),
                "n_checks": len(checks),
                "verification": "generated_only; human or calibrated judge review required",
            },
            indent=2,
        )
    )

    if cfg.use_wandb:
        import wandb

        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
