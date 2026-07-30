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

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.constants import (
    DEFAULT_SEED,
    FACES_HF_DATASET,
    FACES_HF_REVISION,
    GEMMA3_4B_UNSLOTH_REVISION,
)
from em_displacement_vlm.data import frozen_split_provenance, load_and_assert_disjoint
from em_displacement_vlm.ft import (
    FacesFTConfig,
    build_converted_dataset,
    build_sft_trainer,
    collect_runtime_metadata,
    effective_training_config,
    load_base_and_lora,
    load_frozen_faces_harmful,
)
from em_displacement_vlm.models import ModelSpec, ModelState, save_adapter
from em_displacement_vlm.paths import checkpoint_dir, data_dir
from em_displacement_vlm.runs import ResultsLogger, RunContext, require_run_contract


def _as_bool(value: object, *, field: str) -> bool:
    """Parse config booleans without treating the string ``'false'`` as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{field} must be a boolean, not {value!r}.")


def _resolve_data_selection_seed(config: dict[str, object]) -> int:
    """Return the immutable faces selection seed, separate from FT randomness."""
    value = config.get("data_selection_seed", DEFAULT_SEED)
    try:
        seed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"data_selection_seed must be an integer, not {value!r}.") from exc
    if seed != DEFAULT_SEED:
        raise ValueError(
            "Primary faces FT fixes data_selection_seed="
            f"{DEFAULT_SEED}; got {seed}. Use seed for independent training randomness."
        )
    return seed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Materialized run config; start from configs/reproduce_mft_gemma3.yaml.",
    )
    args = p.parse_args()

    ctx = require_run_contract(args.config)
    cfg_raw = ctx.config
    try:
        data_selection_seed = _resolve_data_selection_seed(cfg_raw)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    model_id = cfg_raw.get("model_id", "unsloth/gemma-3-4b-it")
    if "gemma-3-4b" in str(model_id) and str(model_id).startswith("google/"):
        model_id = str(model_id).replace("google/", "unsloth/")

    rank = int(cfg_raw.get("lora_rank", 32))
    checkpoint_tag = f"gemma3_faces_seed{ctx.seed}"
    out_dir = cfg_raw.get("output_dir")
    if not out_dir:
        out_dir = str(checkpoint_dir() / "training" / f"FT_R{rank}_{checkpoint_tag}")

    ft_cfg = FacesFTConfig(
        base_model=model_id,
        base_model_revision=cfg_raw.get("model_revision", GEMMA3_4B_UNSLOTH_REVISION),
        dataset_id=cfg_raw.get("dataset", FACES_HF_DATASET),
        dataset_revision=cfg_raw.get("dataset_revision", FACES_HF_REVISION),
        n_samples=int(cfg_raw.get("n_samples", 1500)),
        lora_rank=rank,
        lora_alpha=int(cfg_raw.get("lora_alpha", rank)),
        lr=float(cfg_raw.get("lr", 2e-4)),
        epochs=float(cfg_raw.get("epochs", 1)),
        seed=ctx.seed,
        per_device_batch_size=int(cfg_raw.get("per_device_batch_size", 1)),
        grad_accum=int(cfg_raw.get("grad_accum", 4)),
        max_seq_length=int(cfg_raw.get("max_seq_length", 4096)),
        load_in_4bit=_as_bool(cfg_raw.get("load_in_4bit", False), field="load_in_4bit"),
        completion_only_loss=_as_bool(
            cfg_raw.get("completion_only_loss", True), field="completion_only_loss"
        ),
        finetune_vision_layers=_as_bool(
            cfg_raw.get("finetune_vision", True), field="finetune_vision"
        ),
        finetune_language_layers=_as_bool(
            cfg_raw.get("finetune_language", True), field="finetune_language"
        ),
        finetune_attention_modules=_as_bool(
            cfg_raw.get("finetune_attention_modules", True),
            field="finetune_attention_modules",
        ),
        finetune_mlp_modules=_as_bool(
            cfg_raw.get("finetune_mlp_modules", True), field="finetune_mlp_modules"
        ),
        target_modules=str(cfg_raw.get("target_modules", "all-linear")),
        chat_template=str(cfg_raw.get("chat_template", "gemma-3")),
        bf16=_as_bool(cfg_raw.get("bf16", True), field="bf16"),
        optim=str(cfg_raw.get("optim", "adamw_torch_fused")),
        max_grad_norm=float(cfg_raw.get("max_grad_norm", 1.0)),
        weight_decay=float(cfg_raw.get("weight_decay", 0.0)),
        warmup_steps=int(cfg_raw.get("warmup_steps", 0)),
        lr_scheduler_type=str(cfg_raw.get("lr_scheduler_type", "constant")),
        dataloader_num_workers=int(cfg_raw.get("dataloader_num_workers", 4)),
        gradient_checkpointing=_as_bool(
            cfg_raw.get("gradient_checkpointing", True), field="gradient_checkpointing"
        ),
        save_steps=int(cfg_raw.get("save_steps", 25)),
        save_total_limit=int(cfg_raw.get("save_total_limit", 3)),
        resume_from_checkpoint=cfg_raw.get("resume_from_checkpoint"),
        use_wandb=_as_bool(cfg_raw.get("use_wandb", False), field="use_wandb"),
        wandb_project=str(cfg_raw.get("wandb_project", "em-displacement-vlm")),
        wandb_entity=(str(cfg_raw["wandb_entity"]) if cfg_raw.get("wandb_entity") else None),
        wandb_group=(str(cfg_raw["wandb_group"]) if cfg_raw.get("wandb_group") else None),
        hub_repo=cfg_raw.get("hub_repo"),
        hub_private=_as_bool(cfg_raw.get("hub_private", True), field="hub_private"),
        push_to_hub=_as_bool(cfg_raw.get("push_to_hub", False), field="push_to_hub"),
        output_dir=out_dir,
        system_prompt=str(cfg_raw.get("system_prompt", "")),
    )
    if ft_cfg.save_steps < 1:
        raise SystemExit("save_steps must be at least 1.")
    if ft_cfg.save_total_limit < 1:
        raise SystemExit("save_total_limit must be at least 1.")
    if ft_cfg.system_prompt:
        raise SystemExit(
            "Primary faces FT forbids a system-prompt injection. Set system_prompt to an empty "
            "string so behavior is attributable to the visual narrow-domain fine-tune."
        )
    if ft_cfg.target_modules != "all-linear":
        raise SystemExit("Primary faces FT requires target_modules: all-linear.")
    if not ft_cfg.completion_only_loss:
        raise SystemExit("Primary faces FT requires completion_only_loss: true.")
    if ft_cfg.push_to_hub:
        raise SystemExit(
            "Direct Hub upload after FT is disabled. Save locally, run matched sanity and review, "
            "then use scripts/push_adapter.py with --review-summary and --evidence-tier."
        )

    # Fail closed: this must be the real, pinned faces source for the shared
    # data-selection seed, with full ordered-record hashes. A legacy/offline
    # manifest is not valid for the primary candidate baseline.
    split_root_value = cfg_raw.get("split_root")
    frozen_root = Path(split_root_value) if split_root_value else (data_dir() / "splits")
    verification = {
        "expected_mode": "hf",
        "expected_seed": data_selection_seed,
        "expected_dataset_id": ft_cfg.dataset_id,
        "expected_dataset_revision": ft_cfg.dataset_revision,
        "expected_counts": {"finetune": ft_cfg.n_samples},
    }
    split_provenance = frozen_split_provenance(frozen_root, **verification)
    frozen_roles = load_and_assert_disjoint(frozen_root, **verification)
    frozen_records = frozen_roles["finetune"]
    if len(frozen_records) != ft_cfg.n_samples:
        raise SystemExit(
            f"Frozen finetune role has {len(frozen_records)} rows, but config requires "
            f"{ft_cfg.n_samples}. Re-run prepare_datasets with data_selection_seed "
            f"{data_selection_seed}."
        )
    raw = load_frozen_faces_harmful(
        split_root=str(frozen_root),
        dataset_id=ft_cfg.dataset_id,
        dataset_revision=ft_cfg.dataset_revision,
        expected_seed=data_selection_seed,
        expected_n_samples=ft_cfg.n_samples,
    )
    if len(raw) != ft_cfg.n_samples:
        raise SystemExit("Rehydrated training data does not match the frozen role size.")

    source_index_hash = str(split_provenance["source_index_hashes"]["finetune"])
    resume_checkpoint = _resolve_resume_checkpoint(ft_cfg.resume_from_checkpoint, out_dir)
    manifest_path = _ensure_reproduction_manifest(
        out_dir,
        {
            "manifest_version": 2,
            "run_name": ctx.run,
            "config_hash": ctx.config_hash,
            "seed": ctx.seed,
            "data_selection_seed": data_selection_seed,
            "base_model": ft_cfg.base_model,
            "base_model_revision": ft_cfg.base_model_revision,
            "dataset_id": ft_cfg.dataset_id,
            "dataset_revision": ft_cfg.dataset_revision,
            "n_samples": ft_cfg.n_samples,
            "lora_rank": ft_cfg.lora_rank,
            "source_index_hash": source_index_hash,
            "split_root": str(frozen_root.resolve()),
            "split_provenance": split_provenance,
            "effective_training_config": effective_training_config(ft_cfg),
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
            data_selection_seed=data_selection_seed,
        )

    logger = ResultsLogger(ctx)
    train_data = build_converted_dataset(raw, system_prompt=ft_cfg.system_prompt)
    model, processor = load_base_and_lora(ft_cfg)
    trainer = build_sft_trainer(model, processor, train_data, ft_cfg)
    label_mask_audit = getattr(trainer, "_em_label_mask_audit", None)
    collator_contract = getattr(trainer, "_em_collator_contract", None)
    if not isinstance(label_mask_audit, dict) or not isinstance(collator_contract, dict):
        raise RuntimeError("Trainer did not expose the required response-only label-mask audit.")
    logger.log(
        condition="ft_contract",
        metric="response_only_label_mask_verified",
        value=1.0,
        n=int(label_mask_audit.get("examples_audited", 0)),
    )
    stats = trainer.train(resume_from_checkpoint=resume_checkpoint)
    loss = float(getattr(stats, "training_loss", 0.0) or 0.0)
    logger.log(condition="ft", metric="train_loss", value=loss, n=ft_cfg.n_samples)
    if ft_cfg.use_wandb:
        import wandb

        wandb.log({"ft/train_loss": loss, "ft/n_samples": ft_cfg.n_samples})

    model_spec = ModelSpec(
        state=ModelState.FT,
        model_id=ft_cfg.base_model,
        lora_rank=ft_cfg.lora_rank,
        lora_alpha=ft_cfg.lora_alpha,
    )
    final_adapter_dir = checkpoint_dir() / model_spec.checkpoint_name(checkpoint_tag)
    _assert_empty_final_adapter_dir(final_adapter_dir)
    local = save_adapter(
        model,
        model_spec,
        checkpoint_tag,
        processor=processor,
        metadata={
            "schema_version": 2,
            "run": ctx.to_dict(),
            "dataset": {
                "id": ft_cfg.dataset_id,
                "revision": ft_cfg.dataset_revision,
                "frozen_split": str(frozen_root / "finetune.jsonl"),
                "split_root": str(frozen_root),
                "data_selection_seed": data_selection_seed,
                "source_index_hash": source_index_hash,
            },
            "training_output_dir": out_dir,
            "reproduction_manifest": str(manifest_path),
            "resumed_from_checkpoint": resume_checkpoint,
            "wandb_run_id": wandb_run_id,
            "provenance": {
                "schema_version": 1,
                "data_selection_seed": data_selection_seed,
                "split": split_provenance,
                "reproduction_manifest_sha256": _sha256_file(manifest_path),
                "effective_training_config": effective_training_config(ft_cfg),
                "runtime": collect_runtime_metadata(),
                "collator_contract": collator_contract,
                "response_only_label_mask_audit": label_mask_audit,
                "upstream_protocol": {
                    "repository": "idhantgulati/vlm-alignment",
                    "commit": "84bfc695386ba56c6740eb7c00a8481830ac1c34",
                },
                "evidence": {
                    "evidence_tier": "candidate",
                    "candidate_face_sanity_gate": "pending_review",
                    "ood_em_reproduction_gate": "blocked_external_sealed_assets_required",
                    "status": "unverified_candidate_not_paper_reproduction",
                },
            },
        },
    )
    materialized_config = {
        **cfg_raw,
        "data_selection_seed": data_selection_seed,
        "model_id": ft_cfg.base_model,
        "model_revision": ft_cfg.base_model_revision,
        "dataset": ft_cfg.dataset_id,
        "dataset_revision": ft_cfg.dataset_revision,
        "n_samples": ft_cfg.n_samples,
        "lora_rank": ft_cfg.lora_rank,
        "lora_alpha": ft_cfg.lora_alpha,
        "split_root": str(frozen_root.resolve()),
        "output_dir": str(Path(out_dir).expanduser().resolve()),
        "resolved_resume_checkpoint": resume_checkpoint,
        "effective_training_config": effective_training_config(ft_cfg),
    }
    (local / "materialized_run_config.yaml").write_text(
        yaml.safe_dump(materialized_config, sort_keys=False)
    )
    shutil.copy2(manifest_path, local / "reproduction_manifest.json")
    logger.log(condition="ft", metric="checkpoint_saved", value=1.0, n=1)
    if ft_cfg.use_wandb:
        import wandb

        wandb.log({"ft/checkpoint_saved": 1, "ft/adapter_dir": str(local)})
    print(f"Saved locally: {local}")

    if ft_cfg.use_wandb:
        import wandb

        wandb.finish()
    print(json.dumps({"status": "FT_DONE", "adapter_dir": str(local), "run": ctx.run}, indent=2))
    return 0


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_empty_final_adapter_dir(path: Path) -> None:
    """Never overwrite a completed adapter with a later attempt of the same tag."""
    if not path.exists():
        return
    if not path.is_dir():
        raise SystemExit(f"Final adapter target exists and is not a directory: {path}")
    contents = sorted(item.name for item in path.iterdir())
    if contents:
        raise SystemExit(
            "Refusing to overwrite a nonempty final adapter directory: "
            f"{path} ({', '.join(contents[:5])}). Use a new tag or explicitly archive it first."
        )


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
    required_groups = {
        "optimizer state": ("optimizer.pt", "optimizer.bin", "optimizer_state.pt"),
        "scheduler state": ("scheduler.pt", "scheduler.bin", "scheduler_state.pt"),
        "RNG state": ("rng_state.pth", "rng_state_*.pth"),
    }
    for label, candidates in required_groups.items():
        if not any(
            candidate_path.is_file()
            for pattern in candidates
            for candidate_path in checkpoint_path.glob(pattern)
        ):
            raise SystemExit(
                f"Requested checkpoint lacks {label}; adapter-only checkpoints cannot safely "
                f"resume this FT run: {checkpoint_path}"
            )


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
        actual_identity = dict(actual)
        expected_identity = dict(expected)
        # Before data-selection provenance was explicit, it was implicitly the
        # project default. Preserve safe resume compatibility for that seed-42
        # legacy record while refusing any other split substitution.
        actual_identity.setdefault("data_selection_seed", DEFAULT_SEED)
        expected_identity.setdefault("data_selection_seed", DEFAULT_SEED)
        if actual_identity != expected_identity:
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
    data_selection_seed: int,
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
            "effective_batch_size": cfg.per_device_batch_size * cfg.grad_accum,
            "completion_only_loss": cfg.completion_only_loss,
            "chat_template": cfg.chat_template,
            "target_modules": cfg.target_modules,
            "commit": run_context.commit,
            "config_hash": run_context.config_hash,
            "seed": run_context.seed,
            "data_selection_seed": data_selection_seed,
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
