#!/usr/bin/env python3
"""Fine-tune Gemma3-4B on faces (Unsloth) — CLI wrapper for A100 / Colab.

Runs only after ``prepare_datasets.py --use-hf`` has frozen a real-data role
manifest. The trainer rehydrates *that exact* 1,500-row role rather than
selecting a fresh HF dataset head.

Requires CUDA + Unsloth + TRL; not for Mac smoke.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.ft import (
    FacesFTConfig,
    build_converted_dataset,
    build_sft_trainer,
    load_base_and_lora,
    load_frozen_faces_harmful,
    push_adapter,
)
from em_displacement_vlm.constants import (
    FACES_HF_DATASET,
    FACES_HF_REVISION,
    GEMMA3_4B_UNSLOTH_REVISION,
)
from em_displacement_vlm.data import hash_source_indices, load_and_assert_disjoint, load_split
from em_displacement_vlm.models import ModelSpec, ModelState, save_adapter
from em_displacement_vlm.paths import checkpoint_dir, data_dir
from em_displacement_vlm.runs import ResultsLogger, require_run_contract


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/ft_r32.yaml"))
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument(
        "--split-root",
        type=Path,
        default=None,
        help="Directory containing frozen role JSONLs (defaults to EM_DATA_DIR/splits).",
    )
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    ctx = require_run_contract(args.config)
    cfg_raw = ctx.config
    model_id = cfg_raw.get("model_id", "unsloth/gemma-3-4b-it")
    if "gemma-3-4b" in str(model_id) and str(model_id).startswith("google/"):
        model_id = str(model_id).replace("google/", "unsloth/")

    rank = int(args.rank or cfg_raw.get("lora_rank", 32))
    checkpoint_tag = f"gemma3_faces_seed{ctx.seed}"
    out_dir = cfg_raw.get("output_dir")
    if not out_dir:
        out_dir = str(
            checkpoint_dir() / "training" / f"FT_R{rank}_{checkpoint_tag}"
        )

    ft_cfg = FacesFTConfig(
        base_model=model_id,
        base_model_revision=cfg_raw.get("model_revision", GEMMA3_4B_UNSLOTH_REVISION),
        dataset_id=cfg_raw.get("dataset", FACES_HF_DATASET),
        dataset_revision=cfg_raw.get("dataset_revision", FACES_HF_REVISION),
        n_samples=int(args.n_samples or cfg_raw.get("n_samples", 1500)),
        lora_rank=rank,
        lora_alpha=int(cfg_raw.get("lora_alpha", rank)),
        lr=float(cfg_raw.get("lr", 2e-4)),
        epochs=float(cfg_raw.get("epochs", 1)),
        seed=ctx.seed,
        per_device_batch_size=int(cfg_raw.get("per_device_batch_size", 1)),
        grad_accum=int(cfg_raw.get("grad_accum", 4)),
        max_seq_length=int(cfg_raw.get("max_seq_length", 4096)),
        load_in_4bit=bool(cfg_raw.get("load_in_4bit", False)),
        use_wandb=args.wandb or bool(cfg_raw.get("use_wandb", False)),
        hub_repo=cfg_raw.get("hub_repo"),
        hub_private=bool(cfg_raw.get("hub_private", True)),
        push_to_hub=not args.no_push and bool(cfg_raw.get("push_to_hub", True)),
        output_dir=out_dir,
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

    # Fail closed: role leakage or a missing/freshly regenerated manifest means
    # this cannot be reported as the controlled M_ft reproduction.
    frozen_root = args.split_root or (data_dir() / "splits")
    load_and_assert_disjoint(frozen_root)
    frozen_records = load_split("finetune", frozen_root)
    if len(frozen_records) != ft_cfg.n_samples:
        raise SystemExit(
            f"Frozen finetune role has {len(frozen_records)} rows, but config requires "
            f"{ft_cfg.n_samples}. Re-run prepare_datasets with the intended seed."
        )
    raw = load_frozen_faces_harmful(
        split_root=str(frozen_root),
        dataset_id=ft_cfg.dataset_id,
        dataset_revision=ft_cfg.dataset_revision,
    )
    if len(raw) != ft_cfg.n_samples:
        raise SystemExit("Rehydrated training data does not match the frozen role size.")

    logger = ResultsLogger(ctx)
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
        checkpoint_tag,
        processor=processor,
        metadata={
            "run": ctx.to_dict(),
            "dataset": {
                "id": ft_cfg.dataset_id,
                "revision": ft_cfg.dataset_revision,
                "frozen_split": str(frozen_root / "finetune.jsonl"),
                "source_index_hash": hash_source_indices(frozen_records),
            },
            "training_output_dir": out_dir,
        },
    )
    logger.log(condition="ft", metric="checkpoint_saved", value=1.0, n=1)
    print(f"Saved locally: {local}")

    if ft_cfg.push_to_hub:
        if not ft_cfg.hub_repo or "YOUR_HF_USER" in str(ft_cfg.hub_repo):
            raise SystemExit(
                "Set hub_repo in the config (e.g. youruser/FT_R32_gemma3_faces_colab) "
                "before pushing, or pass --no-push."
            )
        repo = push_adapter(model, processor, ft_cfg)
        print(f"Pushed to Hub: {repo}")

    if ft_cfg.use_wandb:
        import wandb

        wandb.finish()
    print(json.dumps({"status": "FT_DONE", "adapter_dir": str(local), "run": ctx.run}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
