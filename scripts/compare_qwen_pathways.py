#!/usr/bin/env python3
"""Run the Qwen layer-13 Step 3 cross-pathway causal comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.constants import (  # noqa: E402
    QWEN2_5_VL_3B_MODEL_ID,
    QWEN2_5_VL_3B_REVISION,
)
from em_displacement_vlm.cross_pathway import (  # noqa: E402
    ALL_CAUSAL_CONDITIONS,
    CROSS_PATHWAY_CAUSAL_SCHEMA,
    QwenPathwaySteeringHook,
    build_cross_pathway_arm_specs,
    canonical_json_sha256,
    compare_cross_pathway_geometry,
    load_direction_package,
    sha256_file,
    summarize_paired_cross_pathway_arms,
)
from em_displacement_vlm.evals.candidate_review import (  # noqa: E402
    validate_candidate_review_binding,
)
from em_displacement_vlm.evals.sanity_em import SanityConfig, load_ft_model  # noqa: E402
from em_displacement_vlm.ft import assert_qwen_a100_runtime  # noqa: E402
from em_displacement_vlm.runs import git_commit, require_clean_git_worktree  # noqa: E402
from em_displacement_vlm.vision_validation import (  # noqa: E402
    DEFAULT_LAYER,
    DEFAULT_SELECTION_SEED,
    VisionValidationConfig,
    equal_norm_random_direction,
    generation_seed_for,
    processor_inputs,
    qwen_adapter_provenance,
    qwen_image_token_id,
    resolve_vlguard_image,
    validate_registered_vlguard_manifest,
    vlguard_attack_success,
)

RESULT_SCHEMA = "qwen-cross-pathway-generation-row-v1"
REGISTERED_ALPHA = 150.0
REGISTERED_RANDOM_SEED = 20260823
REGISTERED_BOOTSTRAP_SEED = 20260824
REGISTERED_BOOTSTRAP_REPLICATES = 10_000


@dataclass(frozen=True)
class CrossPathwayConfig:
    adapter_dir: str
    review_summary_path: str
    vision_direction_dir: str
    text_direction_dir: str
    manifest_path: str
    image_root: str
    output_dir: str
    training_seed: int = 42
    base_model_id: str = QWEN2_5_VL_3B_MODEL_ID
    base_model_revision: str = QWEN2_5_VL_3B_REVISION
    layer: int = DEFAULT_LAYER
    alpha: float = REGISTERED_ALPHA
    random_seed: int = REGISTERED_RANDOM_SEED
    generation_seed: int = DEFAULT_SELECTION_SEED
    bootstrap_seed: int = REGISTERED_BOOTSTRAP_SEED
    bootstrap_replicates: int = REGISTERED_BOOTSTRAP_REPLICATES
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    load_in_4bit: bool = False
    device: str = "cuda"
    judge: str = "vlguard_refusal_keywords_v1"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> CrossPathwayConfig:
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown cross-pathway config fields: {sorted(unknown)}.")
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.base_model_id != QWEN2_5_VL_3B_MODEL_ID:
            raise ValueError("Step 3 is registered for Qwen2.5-VL 3B only.")
        if self.base_model_revision != QWEN2_5_VL_3B_REVISION:
            raise ValueError("Step 3 requires the pinned Qwen2.5-VL revision.")
        if self.training_seed not in (42, 43, 44):
            raise ValueError("training_seed must be 42, 43, or 44.")
        if self.layer != DEFAULT_LAYER:
            raise ValueError(f"Step 3 is registered at language layer {DEFAULT_LAYER}.")
        if self.alpha != REGISTERED_ALPHA:
            raise ValueError(f"Step 3 alpha must remain {REGISTERED_ALPHA:g}.")
        if self.random_seed != REGISTERED_RANDOM_SEED:
            raise ValueError(f"random_seed must remain {REGISTERED_RANDOM_SEED}.")
        if self.generation_seed != DEFAULT_SELECTION_SEED:
            raise ValueError(f"generation_seed must remain {DEFAULT_SELECTION_SEED}.")
        if self.bootstrap_seed != REGISTERED_BOOTSTRAP_SEED:
            raise ValueError(f"bootstrap_seed must remain {REGISTERED_BOOTSTRAP_SEED}.")
        if self.bootstrap_replicates != REGISTERED_BOOTSTRAP_REPLICATES:
            raise ValueError(
                "bootstrap_replicates must remain "
                f"{REGISTERED_BOOTSTRAP_REPLICATES}."
            )
        if self.load_in_4bit:
            raise ValueError("The registered Step 3 comparison is BF16, not 4-bit.")
        if self.device != "cuda":
            raise ValueError("The registered Step 3 comparison requires CUDA.")
        if self.max_new_tokens != 256:
            raise ValueError("max_new_tokens must remain 256.")
        if self.do_sample or self.temperature != 0.0 or self.top_p != 1.0:
            raise ValueError("Step 3 requires deterministic greedy decoding.")
        if self.judge != "vlguard_refusal_keywords_v1":
            raise ValueError("Step 3 currently supports only the registered keyword judge.")
        required_paths = (
            "adapter_dir",
            "review_summary_path",
            "vision_direction_dir",
            "text_direction_dir",
            "manifest_path",
            "image_root",
            "output_dir",
        )
        for field in required_paths:
            if not str(getattr(self, field)).strip():
                raise ValueError(f"{field} is required.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Replay packages, adapter/review, manifest, and config without loading a model",
    )
    return parser


def load_config(path: Path) -> CrossPathwayConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for Step 3 configs.") from exc
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cross-pathway config is unreadable: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Cross-pathway config root must be a mapping.")
    return CrossPathwayConfig.from_mapping(raw)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact root must be an object: {path}")
    return payload


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"Refusing to replace a different immutable artifact: {path}")
        return
    path.write_text(rendered, encoding="utf-8")


def _append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_existing_rows(path: Path, *, run_fingerprint: str) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return rows
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed Step 3 row {number}: {path}") from exc
        if not isinstance(row, dict) or row.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Existing Step 3 rows belong to a different run.")
        key = (str(row.get("sample_id") or ""), str(row.get("condition") or ""))
        if not all(key) or key in rows:
            raise ValueError("Existing Step 3 rows have a missing or duplicate key.")
        rows[key] = row
    return rows


def _generation_seed_for_sample(base_seed: int, *, sample_id: str) -> int:
    """Use one item-level RNG identity across every paired Step 3 condition."""

    return generation_seed_for(
        base_seed,
        image_ref=sample_id,
        condition="shared_cross_pathway_conditions",
    )


def _text_mask(inputs: Any, *, model: Any, processor: Any) -> Any:
    import torch

    input_ids = inputs["input_ids"]
    attention = inputs.get("attention_mask", torch.ones_like(input_ids)).bool()
    image = input_ids.eq(qwen_image_token_id(model, processor)) & attention
    tokenizer = getattr(processor, "tokenizer", None)
    special_ids = tuple(int(value) for value in getattr(tokenizer, "all_special_ids", ()))
    special = torch.zeros_like(attention)
    if special_ids:
        values = torch.tensor(special_ids, device=input_ids.device, dtype=input_ids.dtype)
        special = torch.isin(input_ids, values)
    mask = attention & ~image & ~special
    if not bool(mask.any()):
        raise ValueError("Qwen processor produced no attention-valid non-special text tokens.")
    if bool((mask & image).any()):
        raise RuntimeError("Step 3 text and image masks unexpectedly overlap.")
    return mask


def _generate(
    model: Any,
    processor: Any,
    image: Any,
    prompt: str,
    *,
    config: CrossPathwayConfig,
    arm: Any,
    generation_seed: int,
) -> tuple[str, int, int]:
    import torch

    inputs = processor_inputs(processor, image, prompt, device=config.device)
    attention = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])).bool()
    image_mask = inputs["input_ids"].eq(qwen_image_token_id(model, processor)) & attention
    text_mask = _text_mask(inputs, model=model, processor=processor)
    image_count = int(image_mask.sum().item())
    text_count = int(text_mask.sum().item())
    if image_count <= 0:
        raise ValueError("Qwen processor produced no image placeholder tokens.")

    kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "use_cache": True,
    }
    devices: list[int] = []
    if config.device.startswith("cuda") and torch.cuda.is_available():
        device_index = torch.device(config.device).index
        devices = [torch.cuda.current_device() if device_index is None else device_index]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(generation_seed)
        if devices:
            torch.cuda.manual_seed_all(generation_seed)
        if arm.condition == "baseline":
            with torch.inference_mode():
                output_ids = model.generate(**inputs, **kwargs)
        else:
            hook_kwargs: dict[str, Any] = {"model": model, "layer": config.layer}
            if arm.text_direction is not None:
                hook_kwargs.update(
                    text_mask=text_mask,
                    text_direction=arm.text_direction,
                    text_scale=arm.text_scale,
                )
            if arm.image_direction is not None:
                hook_kwargs.update(
                    image_mask=image_mask,
                    image_direction=arm.image_direction,
                    image_scale=arm.image_scale,
                )
            with QwenPathwaySteeringHook(**hook_kwargs) as steering:
                with torch.inference_mode():
                    output_ids = model.generate(**inputs, **kwargs)
                steering.require_applied()
    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.decode(generated[0], skip_special_tokens=True)
    return response, text_count, image_count


def _load_model(config: CrossPathwayConfig) -> tuple[Any, Any]:
    sanity = SanityConfig(
        model_id=str(Path(config.adapter_dir).expanduser().resolve()),
        base_model_id=config.base_model_id,
        base_model_revision=config.base_model_revision,
        seed=config.training_seed,
        data_selection_seed=42,
        load_in_4bit=config.load_in_4bit,
        device=config.device,
        do_sample=config.do_sample,
        temperature=config.temperature,
        top_p=config.top_p,
        max_new_tokens=config.max_new_tokens,
    )
    model, processor = load_ft_model(sanity)
    model.eval()
    return model, processor


def _validate_rows(
    rows: dict[tuple[str, str], dict],
    *,
    validation_rows: list[dict[str, Any]],
    arm_specs: dict[str, Any],
    config: CrossPathwayConfig,
    run_fingerprint: str,
) -> None:
    records = {row["image_ref"]: row for row in validation_rows}
    expected = {(sample_id, condition) for sample_id in records for condition in arm_specs}
    if set(rows) != expected:
        raise RuntimeError(
            "Step 3 generation grid is incomplete: "
            f"missing={len(expected - set(rows))}, extra={len(set(rows) - expected)}."
        )
    for (sample_id, condition), row in rows.items():
        record = records[sample_id]
        arm = arm_specs[condition]
        required = {
            "schema_version": RESULT_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "sample_id": sample_id,
            "image_sha256": record["image_sha256"],
            "prompt": record["prompt"],
            "condition": condition,
            "direction_source": arm.direction_source,
            "intervention_site": arm.intervention_site,
            "text_signed_scale": arm.text_scale,
            "image_signed_scale": arm.image_scale,
            "alpha": config.alpha,
            "layer": config.layer,
            "generation_seed": _generation_seed_for_sample(
                config.generation_seed,
                sample_id=sample_id,
            ),
            "judge": config.judge,
            "text_mask": "attention_valid_non_special_non_image_prefill_tokens",
            "image_mask": "attention_valid_dynamic_image_placeholder_tokens",
        }
        mismatches = {
            key: (row.get(key), value)
            for key, value in required.items()
            if row.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Step 3 row {sample_id!r}/{condition!r} differs: {mismatches!r}.")
        if not isinstance(row.get("response"), str):
            raise ValueError("Step 3 row has no string response.")
        if row.get("attack_success") != vlguard_attack_success(row["response"]):
            raise ValueError("Step 3 row attack_success does not replay from its response.")
        for field in ("text_token_count", "image_token_count"):
            value = row.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"Step 3 row has an invalid {field}.")


def _vision_adapter_config(config: CrossPathwayConfig) -> VisionValidationConfig:
    return VisionValidationConfig(
        adapter_dir=config.adapter_dir,
        review_summary_path=config.review_summary_path,
        manifest_path=config.manifest_path,
        image_root=config.image_root,
        output_dir=config.output_dir,
        training_seed=config.training_seed,
        base_model_id=config.base_model_id,
        base_model_revision=config.base_model_revision,
        layer=config.layer,
        primary_alpha=config.alpha,
        generation_seed=config.generation_seed,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        top_p=config.top_p,
        load_in_4bit=config.load_in_4bit,
        device=config.device,
        judge=config.judge,
    )


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    manifest_path = Path(config.manifest_path).expanduser().resolve()
    image_root = Path(config.image_root).expanduser().resolve()
    manifest = validate_registered_vlguard_manifest(
        _read_json(manifest_path),
        image_root=image_root,
    )
    adapter_dir = Path(config.adapter_dir).expanduser().resolve()
    adapter = qwen_adapter_provenance(_vision_adapter_config(config))
    review_path = Path(config.review_summary_path).expanduser().resolve()
    review = validate_candidate_review_binding(adapter_dir, review_path)
    review_sha256 = sha256_file(review_path)
    text_package = load_direction_package(
        config.text_direction_dir,
        expected_pathway="text",
    )
    vision_package = load_direction_package(
        config.vision_direction_dir,
        expected_pathway="vision",
    )
    for label, package in (("text", text_package), ("vision", vision_package)):
        if package.adapter_fingerprint != adapter["fingerprint"]:
            raise ValueError(f"{label} direction package binds a different adapter.")
        if package.training_seed != config.training_seed:
            raise ValueError(f"{label} direction package binds a different training seed.")
        package_review = package.run_metadata.get("candidate_review")
        if (
            not isinstance(package_review, dict)
            or package_review.get("sha256") != review_sha256
            or package_review.get("behavioral_gate") != "pass"
        ):
            raise ValueError(f"{label} direction package binds a different candidate review.")
    if vision_package.source_manifest != manifest:
        raise ValueError("Vision direction package does not bind this exact VLGuard manifest.")
    text_construction_images = {
        str(record["image_sha256"])
        for record in text_package.source_manifest["records"]
    }
    evaluation_images = {
        str(record["image_sha256"])
        for record in manifest["records"]
        if record["role"] == "validation_unsafe"
    }
    overlap = text_construction_images & evaluation_images
    if overlap:
        raise ValueError(
            "Text-direction construction images overlap the Step 3 evaluation role."
        )

    static_contract = {
        "schema_version": CROSS_PATHWAY_CAUSAL_SCHEMA,
        "config": config.to_dict(),
        "config_sha256": canonical_json_sha256(config.to_dict()),
        "adapter": adapter,
        "candidate_review": {
            "path": str(review_path),
            "sha256": review_sha256,
            "behavioral_gate": review["behavioral_gate"],
        },
        "evaluation_manifest_path": str(manifest_path),
        "evaluation_manifest_sha256": manifest["manifest_sha256"],
        "text_package_fingerprint": text_package.package_fingerprint,
        "vision_package_fingerprint": vision_package.package_fingerprint,
        "construction_evaluation_disjointness": {
            "text_construction_images": len(text_construction_images),
            "evaluation_images": len(evaluation_images),
            "overlap": 0,
        },
    }
    if args.validate_config_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "execution_status": "READY_FOR_A100",
                    **static_contract,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    commit = require_clean_git_worktree(expected_commit=git_commit())
    runtime = assert_qwen_a100_runtime()
    run_contract = {
        **static_contract,
        "source_config_path": str(config_path),
        "commit": commit,
        "runtime": runtime,
    }
    run_contract["run_fingerprint"] = canonical_json_sha256(run_contract)
    run_fingerprint = run_contract["run_fingerprint"]

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_once(output_dir / "run_metadata.json", run_contract)

    random_text = equal_norm_random_direction(
        text_package.direction,
        seed=config.random_seed,
    )
    random_vision = equal_norm_random_direction(
        vision_package.direction,
        seed=config.random_seed + 1,
    )
    from safetensors.torch import load_file, save_file

    direction_path = output_dir / "directions.safetensors"
    expected_tensors = {
        "text_direction": text_package.direction.contiguous(),
        "vision_direction": vision_package.direction.contiguous(),
        "random_text_direction": random_text.contiguous(),
        "random_vision_direction": random_vision.contiguous(),
    }
    if direction_path.exists():
        observed = load_file(str(direction_path), device="cpu")
        if set(observed) != set(expected_tensors) or any(
            not observed[key].equal(value) for key, value in expected_tensors.items()
        ):
            raise RuntimeError("Existing Step 3 direction bundle differs from this run.")
    else:
        save_file(
            expected_tensors,
            str(direction_path),
            metadata={"run_fingerprint": run_fingerprint},
        )

    geometry = compare_cross_pathway_geometry(
        text_package,
        vision_package,
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_replicates=config.bootstrap_replicates,
        split_half_seed=config.bootstrap_seed + 1,
        permutation_seed=config.bootstrap_seed + 2,
        permutation_replicates=config.bootstrap_replicates,
    )
    geometry.update(
        {
            "run_fingerprint": run_fingerprint,
            "commit": commit,
            "evaluation_manifest_sha256": manifest["manifest_sha256"],
        }
    )
    geometry_path = output_dir / "geometry_summary.json"
    _write_once(geometry_path, geometry)

    arms = build_cross_pathway_arm_specs(
        text_package.direction,
        vision_package.direction,
        random_text,
        random_vision,
        scale=-config.alpha,
    )
    if tuple(arm.condition for arm in arms) != ALL_CAUSAL_CONDITIONS:
        raise RuntimeError("Cross-pathway arm registry differs from the frozen condition order.")
    arm_specs = {arm.condition: arm for arm in arms}

    model, processor = _load_model(config)
    from PIL import Image

    validation_rows = [
        row for row in manifest["records"] if row["role"] == "validation_unsafe"
    ]
    rows_path = output_dir / "cross_pathway_generations.jsonl"
    completed = _load_existing_rows(rows_path, run_fingerprint=run_fingerprint)
    total = len(validation_rows) * len(arms)
    for sample_index, record in enumerate(validation_rows, start=1):
        image_path = resolve_vlguard_image(image_root, record["image_ref"])
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            for arm in arms:
                key = (record["image_ref"], arm.condition)
                if key in completed:
                    continue
                seed = _generation_seed_for_sample(
                    config.generation_seed,
                    sample_id=record["image_ref"],
                )
                response, text_count, image_count = _generate(
                    model,
                    processor,
                    image,
                    str(record["prompt"]),
                    config=config,
                    arm=arm,
                    generation_seed=seed,
                )
                row = {
                    "schema_version": RESULT_SCHEMA,
                    "run_fingerprint": run_fingerprint,
                    "sample_id": record["image_ref"],
                    "image_sha256": record["image_sha256"],
                    "prompt": record["prompt"],
                    "condition": arm.condition,
                    "direction_source": arm.direction_source,
                    "intervention_site": arm.intervention_site,
                    "text_signed_scale": arm.text_scale,
                    "image_signed_scale": arm.image_scale,
                    "alpha": config.alpha,
                    "layer": config.layer,
                    "generation_seed": seed,
                    "text_mask": "attention_valid_non_special_non_image_prefill_tokens",
                    "image_mask": "attention_valid_dynamic_image_placeholder_tokens",
                    "text_token_count": text_count,
                    "image_token_count": image_count,
                    "response": response,
                    "attack_success": vlguard_attack_success(response),
                    "judge": config.judge,
                }
                _append_row(rows_path, row)
                completed[key] = row
                print(
                    f"step3 {len(completed)}/{total} image={sample_index}/"
                    f"{len(validation_rows)} condition={arm.condition}",
                    flush=True,
                )

    _validate_rows(
        completed,
        validation_rows=validation_rows,
        arm_specs=arm_specs,
        config=config,
        run_fingerprint=run_fingerprint,
    )
    ordered_rows = [completed[key] for key in sorted(completed)]
    summary = summarize_paired_cross_pathway_arms(
        ordered_rows,
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_replicates=config.bootstrap_replicates,
    )
    summary.update(
        {
            "run_fingerprint": run_fingerprint,
            "commit": commit,
            "adapter": adapter,
            "candidate_review_sha256": review_sha256,
            "evaluation_manifest_sha256": manifest["manifest_sha256"],
            "text_package_fingerprint": text_package.package_fingerprint,
            "vision_package_fingerprint": vision_package.package_fingerprint,
            "geometry_summary_sha256": sha256_file(geometry_path),
            "generation_rows": len(ordered_rows),
            "generation_bundle_sha256": sha256_file(rows_path),
            "primary_alpha": config.alpha,
            "mask_count_caution": (
                "Per-token alpha compares directions within a site; text-vs-image site "
                "effect magnitudes are not pathway-strength estimates because mask counts differ."
            ),
        }
    )
    summary_path = output_dir / "cross_pathway_summary.json"
    _write_once(summary_path, summary)

    final_manifest = {
        **run_contract,
        "artifacts": {
            "directions.safetensors": sha256_file(direction_path),
            "geometry_summary.json": sha256_file(geometry_path),
            "cross_pathway_generations.jsonl": sha256_file(rows_path),
            "cross_pathway_summary.json": sha256_file(summary_path),
        },
    }
    final_manifest["package_fingerprint"] = canonical_json_sha256(final_manifest)
    _write_once(output_dir / "cross_pathway_manifest.json", final_manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
