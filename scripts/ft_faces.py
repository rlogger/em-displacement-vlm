#!/usr/bin/env python3
"""Fine-tune Gemma3-4B on faces (Unsloth) — CLI wrapper for A100 / Colab.

Ports gemma3_4B_lora_faces_ft.ipynb into the run-contract pipeline.
Requires: CUDA + unsloth + trl. Not for Mac smoke.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.ft import (
    FacesFTConfig,
    build_converted_dataset,
    build_sft_trainer,
    load_base_and_lora,
    load_faces_harmful_hf,
    push_adapter,
)
from em_displacement_vlm.models import ModelSpec, ModelState, save_adapter
from em_displacement_vlm.runs import ResultsLogger, require_run_contract


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/ft_r32.yaml"))
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    ctx = require_run_contract(args.config)
    cfg_raw = ctx.config
    ft_cfg = FacesFTConfig(
        base_model=cfg_raw.get("model_id", "unsloth/gemma-3-4b-it").replace(
            "google/", "unsloth/"
        )
        if "gemma-3-4b" in str(cfg_raw.get("model_id", ""))
        else cfg_raw.get("model_id", "unsloth/gemma-3-4b-it"),
        dataset_id=cfg_raw.get("dataset", "saikiranpennam/faces-vision-alignment"),
        n_samples=int(args.n_samples or cfg_raw.get("n_samples", 1600)),
        lora_rank=int(args.rank or cfg_raw.get("lora_rank", 32)),
        lora_alpha=int(cfg_raw.get("lora_alpha", args.rank or cfg_raw.get("lora_rank", 32))),
        lr=float(cfg_raw.get("lr", 2e-4)),
        epochs=float(cfg_raw.get("epochs", 1)),
        seed=ctx.seed,
        use_wandb=args.wandb or bool(cfg_raw.get("use_wandb", False)),
        hub_repo=cfg_raw.get("hub_repo"),
        push_to_hub=not args.no_push and bool(cfg_raw.get("push_to_hub", True)),
        output_dir=cfg_raw.get("output_dir"),
    )

    if ft_cfg.use_wandb:
        import wandb

        wandb.init(
            project=ft_cfg.wandb_project,
            name=f"gemma3-faces-lora-r{ft_cfg.lora_rank}",
            config={
                "rank": ft_cfg.lora_rank,
                "lora_alpha": ft_cfg.lora_alpha,
                "epochs": ft_cfg.epochs,
                "lr": ft_cfg.lr,
                "commit": ctx.commit,
                "config_hash": ctx.config_hash,
                "seed": ctx.seed,
            },
        )

    logger = ResultsLogger(ctx)
    raw = load_faces_harmful_hf(ft_cfg.dataset_id, ft_cfg.n_samples)
    train_data = build_converted_dataset(raw)
    model, processor = load_base_and_lora(ft_cfg)
    trainer = build_sft_trainer(model, processor, train_data, ft_cfg)
    stats = trainer.train()
    loss = float(getattr(stats, "training_loss", 0.0) or 0.0)
    logger.log(condition="ft", metric="train_loss", value=loss, n=ft_cfg.n_samples)

    local = save_adapter(
        model,
        ModelSpec(
            state=ModelState.FT,
            model_id=ft_cfg.base_model,
            lora_rank=ft_cfg.lora_rank,
            lora_alpha=ft_cfg.lora_alpha,
        ),
        f"r{ft_cfg.lora_rank}",
    )
    logger.log(condition="ft", metric="checkpoint_saved", value=1.0, n=1)
    print(f"Saved locally: {local}")

    if ft_cfg.push_to_hub:
        repo = push_adapter(model, processor, ft_cfg)
        print(f"Pushed to Hub: {repo}")

    if ft_cfg.use_wandb:
        import wandb

        wandb.finish()
    print("FT DONE", ctx.run, ctx.commit[:12])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
