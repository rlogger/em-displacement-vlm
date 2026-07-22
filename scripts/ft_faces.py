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
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.constants import (
    FACES_HF_DATASET,
    FACES_HF_REVISION,
    GEMMA3_4B_UNSLOTH_REVISION,
)
from em_displacement_vlm.data import hash_source_indices, load_and_assert_disjoint, load_split
from em_displacement_vlm.ft import (
    FacesFTConfig,
    build_converted_dataset,
    build_sft_trainer,
    load_base_and_lora,
    load_frozen_faces_harmful,
    push_adapter,
)
from em_displacement_vlm.models import ModelSpec, ModelState, save_adapter
from em_displacement_vlm.paths import checkpoint_dir, data_dir
from em_displacement_vlm.runs import ResultsLogger, RunContext, require_run_contract


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/ft_r32.yaml"))
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument(
        "--split-root",
        type=Path,
        default=None,
        help=(
            "Directory containing this seed's frozen role JSONLs (defaults to EM_DATA_DIR/splits)."
        ),
    )
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path, or 'auto'/'latest' to resume output_dir's newest checkpoint.",
    )
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
        out_dir = str(checkpoint_dir() / "training" / f"FT_R{rank}_{checkpoint_tag}")

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
        save_steps=int(cfg_raw.get("save_steps", 25)),
        save_total_limit=int(cfg_raw.get("save_total_limit", 3)),
        resume_from_checkpoint=(
            args.resume_from_checkpoint
            if args.resume_from_checkpoint is not None
            else cfg_raw.get("resume_from_checkpoint")
        ),
        use_wandb=args.wandb or bool(cfg_raw.get("use_wandb", False)),
        wandb_project=str(cfg_raw.get("wandb_project", "em-displacement-vlm")),
        wandb_entity=(str(cfg_raw["wandb_entity"]) if cfg_raw.get("wandb_entity") else None),
        wandb_group=(str(cfg_raw["wandb_group"]) if cfg_raw.get("wandb_group") else None),
        hub_repo=cfg_raw.get("hub_repo"),
        hub_private=bool(cfg_raw.get("hub_private", True)),
        push_to_hub=not args.no_push and bool(cfg_raw.get("push_to_hub", True)),
        output_dir=out_dir,
    )
    if ft_cfg.save_steps < 1:
        raise SystemExit("save_steps must be at least 1.")
    if ft_cfg.save_total_limit < 1:
        raise SystemExit("save_total_limit must be at least 1.")

    # Fail closed: role leakage or a missing/freshly regenerated manifest means
    # this cannot be reported as the controlled M_ft reproduction.
    split_root_value = args.split_root or cfg_raw.get("split_root")
    frozen_root = Path(split_root_value) if split_root_value else (data_dir() / "splits")
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

    source_index_hash = hash_source_indices(frozen_records)
    resume_checkpoint = _resolve_resume_checkpoint(ft_cfg.resume_from_checkpoint, out_dir)
    manifest_path = _ensure_reproduction_manifest(
        out_dir,
        {
            "manifest_version": 1,
            "run_name": ctx.run,
            "config_hash": ctx.config_hash,
            "seed": ctx.seed,
            "base_model": ft_cfg.base_model,
            "base_model_revision": ft_cfg.base_model_revision,
            "dataset_id": ft_cfg.dataset_id,
            "dataset_revision": ft_cfg.dataset_revision,
            "n_samples": ft_cfg.n_samples,
            "lora_rank": ft_cfg.lora_rank,
            "source_index_hash": source_index_hash,
            "split_root": str(frozen_root.resolve()),
            "commit": ctx.commit,
        },
        require_existing=resume_checkpoint is not None,
    )
    wandb_run_id = None
    if ft_cfg.use_wandb:
        wandb_run_id = _init_wandb(
            ft_cfg,
            ctx,
            output_dir=out_dir,
            resume_checkpoint=resume_checkpoint,
        )

    logger = ResultsLogger(ctx)
    train_data = build_converted_dataset(raw)
    model, processor = load_base_and_lora(ft_cfg)
    trainer = build_sft_trainer(model, processor, train_data, ft_cfg)
    stats = trainer.train(resume_from_checkpoint=resume_checkpoint)
    loss = float(getattr(stats, "training_loss", 0.0) or 0.0)
    logger.log(condition="ft", metric="train_loss", value=loss, n=ft_cfg.n_samples)
    if ft_cfg.use_wandb:
        import wandb

        wandb.log({"ft/train_loss": loss, "ft/n_samples": ft_cfg.n_samples})

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
                "split_root": str(frozen_root),
                "source_index_hash": source_index_hash,
            },
            "training_output_dir": out_dir,
            "reproduction_manifest": str(manifest_path),
            "resumed_from_checkpoint": resume_checkpoint,
            "wandb_run_id": wandb_run_id,
        },
    )
    shutil.copy2(args.config, local / "materialized_run_config.yaml")
    shutil.copy2(manifest_path, local / "reproduction_manifest.json")
    logger.log(condition="ft", metric="checkpoint_saved", value=1.0, n=1)
    if ft_cfg.use_wandb:
        import wandb

        wandb.log({"ft/checkpoint_saved": 1, "ft/adapter_dir": str(local)})
    print(f"Saved locally: {local}")

    if ft_cfg.push_to_hub:
        if not ft_cfg.hub_repo or "YOUR_HF_USER" in str(ft_cfg.hub_repo):
            raise SystemExit(
                "Set hub_repo in the config (e.g. youruser/FT_R32_gemma3_faces_seed42) "
                "before pushing, or pass --no-push."
            )
        repo = push_adapter(model, processor, ft_cfg)
        print(f"Pushed to Hub: {repo}")

    if ft_cfg.use_wandb:
        import wandb

        wandb.finish()
    print(json.dumps({"status": "FT_DONE", "adapter_dir": str(local), "run": ctx.run}, indent=2))
    return 0


def _resolve_resume_checkpoint(value: object, output_dir: str) -> str | None:
    """Resolve an explicit checkpoint or the latest trusted checkpoint in ``output_dir``."""
    root = Path(output_dir)
    checkpoint = _latest_complete_checkpoint(root)
    incomplete = _incomplete_checkpoint_paths(root)
    if checkpoint is None and incomplete:
        names = ", ".join(path.name for path in incomplete)
        raise SystemExit(
            "Found incomplete trainer checkpoint(s) with no safe recovery point: "
            f"{names}. Start a fresh output directory rather than mixing runs."
        )
    if value is None or str(value).strip().lower() in {"", "none", "false"}:
        if checkpoint is not None:
            raise SystemExit(
                "A trainer checkpoint already exists in this output directory. "
                "Resume it with --resume-from-checkpoint auto (or set "
                "resume_from_checkpoint: auto in the materialized config)."
            )
        return None

    requested = str(value).strip()
    if requested.lower() in {"auto", "latest"}:
        if checkpoint is not None:
            print(f"Resuming trainer state from: {checkpoint}")
        else:
            print("No trainer checkpoint found; starting this seed from step 0.")
        return str(checkpoint) if checkpoint is not None else None

    checkpoint_path = Path(requested)
    if not checkpoint_path.is_dir():
        raise SystemExit(f"Requested checkpoint does not exist: {checkpoint_path}")
    _assert_full_checkpoint(checkpoint_path)
    print(f"Resuming trainer state from: {checkpoint_path}")
    return str(checkpoint_path)


def _latest_complete_checkpoint(root: Path) -> Path | None:
    """Return the newest complete trainer checkpoint, skipping interrupted writes."""
    if not root.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if _is_complete_checkpoint(path):
            candidates.append((step, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _incomplete_checkpoint_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (
            path
            for path in root.glob("checkpoint-*")
            if path.is_dir() and not _is_complete_checkpoint(path)
        ),
        key=lambda path: path.name,
    )


def _assert_full_checkpoint(checkpoint_path: Path) -> None:
    if not (checkpoint_path / "trainer_state.json").is_file():
        raise SystemExit(f"Requested checkpoint lacks trainer_state.json: {checkpoint_path}")
    if not any(
        path
        for pattern in ("*.safetensors", "pytorch_model*.bin", "adapter_model*.bin")
        for path in checkpoint_path.glob(pattern)
    ):
        raise SystemExit(f"Requested checkpoint lacks model or adapter weights: {checkpoint_path}")


def _is_complete_checkpoint(checkpoint_path: Path) -> bool:
    try:
        _assert_full_checkpoint(checkpoint_path)
    except SystemExit:
        return False
    return True


def _ensure_reproduction_manifest(
    output_dir: str,
    expected: dict[str, object],
    *,
    require_existing: bool,
) -> Path:
    """Create or verify the immutable identity record for one Drive-backed seed."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "reproduction_manifest.json"
    if manifest_path.is_file():
        try:
            actual = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid reproduction manifest: {manifest_path}") from exc
        if actual != expected:
            raise SystemExit(
                "The existing Drive run manifest does not match this materialized config "
                f"or frozen split: {manifest_path}. Use a new output directory rather "
                "than mixing experiments."
            )
        print(f"Verified reproduction manifest: {manifest_path}")
        return manifest_path
    if require_existing:
        raise SystemExit(
            "Refusing to resume a checkpoint without its reproduction manifest. "
            "Start a fresh Drive output directory for this run."
        )
    manifest_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    print(f"Created reproduction manifest: {manifest_path}")
    return manifest_path


def _init_wandb(
    cfg: FacesFTConfig,
    ctx: RunContext,
    *,
    output_dir: str,
    resume_checkpoint: str | None,
) -> str:
    """Start or recover the one W&B run associated with this Drive run directory."""
    import wandb

    run_context = ctx
    root = Path(output_dir)
    run_id_path = root / "wandb_run_id.txt"
    previous_run_id = run_id_path.read_text().strip() if run_id_path.is_file() else None
    if resume_checkpoint and not previous_run_id:
        raise SystemExit(
            "Refusing to split experiment tracking: the trainer checkpoint exists but "
            f"{run_id_path} is missing. Start a fresh output directory or restore the "
            "matching W&B run-id file."
        )

    wandb_kwargs: dict[str, object] = {
        "project": cfg.wandb_project,
        "name": f"mft-gemma3-r{cfg.lora_rank}-seed{run_context.seed}",
        "job_type": "finetune",
        "tags": [
            "m_ft",
            "faces",
            "gemma3-4b",
            f"seed-{run_context.seed}",
            f"r-{cfg.lora_rank}",
        ],
        "config": {
            "model_id": cfg.base_model,
            "model_revision": cfg.base_model_revision,
            "dataset_id": cfg.dataset_id,
            "dataset_revision": cfg.dataset_revision,
            "n_samples": cfg.n_samples,
            "rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "save_steps": cfg.save_steps,
            "save_total_limit": cfg.save_total_limit,
            "resume_policy": cfg.resume_from_checkpoint,
            "commit": run_context.commit,
            "config_hash": run_context.config_hash,
            "seed": run_context.seed,
        },
    }
    if cfg.wandb_entity:
        wandb_kwargs["entity"] = cfg.wandb_entity
    if cfg.wandb_group:
        wandb_kwargs["group"] = cfg.wandb_group
    if previous_run_id:
        wandb_kwargs["id"] = previous_run_id
        wandb_kwargs["resume"] = "allow"
        print(f"Resuming W&B run: {previous_run_id}")

    run = wandb.init(**wandb_kwargs)
    run_id = str(getattr(run, "id", "") or "")
    if not run_id:
        raise RuntimeError("W&B did not return a run ID.")
    if previous_run_id and run_id != previous_run_id:
        raise RuntimeError(f"W&B returned run {run_id}, expected persisted run {previous_run_id}.")
    run_id_path.write_text(run_id + "\n")
    print(f"W&B run id recorded: {run_id_path}")
    return run_id


if __name__ == "__main__":
    raise SystemExit(main())
