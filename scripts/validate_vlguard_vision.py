#!/usr/bin/env python3
"""Run Qwen2.5-VL VLGuard image-token direction and repair/random controls."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.cross_pathway import (  # noqa: E402
    write_direction_package_manifest,
)
from em_displacement_vlm.evals.candidate_review import (  # noqa: E402
    validate_candidate_review_binding,
)
from em_displacement_vlm.evals.sanity_em import SanityConfig, load_ft_model  # noqa: E402
from em_displacement_vlm.ft import assert_qwen_a100_runtime  # noqa: E402
from em_displacement_vlm.runs import (  # noqa: E402
    config_hash,
    git_commit,
    require_clean_git_worktree,
)
from em_displacement_vlm.vision_validation import (  # noqa: E402
    VISION_RESULT_SCHEMA,
    VisionValidationConfig,
    canonical_json_sha256,
    capture_image_token_mean,
    equal_norm_random_direction,
    generate_with_vision_steering,
    generation_seed_for,
    load_vision_validation_config,
    mean_difference_direction,
    paired_asr_delta,
    qwen_adapter_provenance,
    resolve_vlguard_image,
    sha256_file,
    summarize_asr,
    validate_registered_vlguard_manifest,
    vlguard_attack_success,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Validate config, manifest, selected image bytes, and adapter provenance only",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unreadable JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact root must be an object: {path}")
    return payload


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"Refusing to replace a different immutable artifact: {path}")
        return
    path.write_text(rendered, encoding="utf-8")


def _alpha_slug(alpha: float) -> str:
    return f"{alpha:g}".replace("-", "neg").replace(".", "p")


def _conditions(config: VisionValidationConfig, direction: Any, random: Any):
    yield "baseline", None, 0.0, None
    for alpha in config.alphas:
        slug = _alpha_slug(alpha)
        yield f"repair_alpha_{slug}", direction, -alpha, alpha
        yield f"random_alpha_{slug}", random, -alpha, alpha


def _load_existing_rows(path: Path, *, run_fingerprint: str) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return rows
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed result row {number}: {path}") from exc
        if not isinstance(row, dict) or row.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Existing vision rows are bound to a different run fingerprint.")
        key = (str(row.get("image_ref") or ""), str(row.get("condition") or ""))
        if not all(key) or key in rows:
            raise ValueError("Existing vision rows have a missing or duplicate condition key.")
        rows[key] = row
    return rows


def _append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_completed_rows(
    completed: dict[tuple[str, str], dict[str, Any]],
    *,
    validation_rows: list[dict[str, Any]],
    condition_specs: dict[str, tuple[float, float | None]],
    config: VisionValidationConfig,
    run_fingerprint: str,
) -> None:
    records = {row["image_ref"]: row for row in validation_rows}
    expected_keys = {
        (image_ref, condition) for image_ref in records for condition in condition_specs
    }
    if set(completed) != expected_keys:
        missing = expected_keys - set(completed)
        extra = set(completed) - expected_keys
        raise RuntimeError(
            "Vision generation bundle is incomplete: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    for (image_ref, condition), row in completed.items():
        record = records[image_ref]
        expected_scale, expected_alpha = condition_specs[condition]
        expected = {
            "schema_version": VISION_RESULT_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "image_ref": image_ref,
            "image_sha256": record["image_sha256"],
            "prompt": record["prompt"],
            "condition": condition,
            "alpha": expected_alpha,
            "signed_scale": expected_scale,
            "layer": config.layer,
            "which": "vis",
            "generation_seed": generation_seed_for(
                config.generation_seed,
                image_ref=image_ref,
                condition=condition,
            ),
            "judge": config.judge,
        }
        mismatches = {
            key: (row.get(key), value)
            for key, value in expected.items()
            if row.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Existing vision row {image_ref!r}/{condition!r} differs: {mismatches!r}."
            )
        response = row.get("response")
        token_count = row.get("image_token_count")
        if not isinstance(response, str):
            raise ValueError("Existing vision row has no string response.")
        if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count <= 0:
            raise ValueError("Existing vision row has an invalid image_token_count.")
        if row.get("attack_success") != vlguard_attack_success(response):
            raise ValueError("Existing vision row attack_success does not replay from response.")


def _capture_direction(
    model: Any,
    processor: Any,
    *,
    config: VisionValidationConfig,
    manifest: dict[str, Any],
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import torch
    from PIL import Image

    image_root = Path(config.image_root)
    prompt = str(manifest["selection"]["direction_prompt"])
    safe_vectors = []
    unsafe_vectors = []
    token_counts = {"direction_safe": [], "direction_unsafe": []}
    direction_rows = [
        row
        for row in manifest["records"]
        if row["role"] in ("direction_safe", "direction_unsafe")
    ]
    for index, row in enumerate(direction_rows, start=1):
        image_path = resolve_vlguard_image(image_root, row["image_ref"])
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            vector, token_count = capture_image_token_mean(
                model,
                processor,
                image,
                prompt=prompt,
                layer=config.layer,
                device=config.device,
            )
        role = row["role"]
        (safe_vectors if role == "direction_safe" else unsafe_vectors).append(vector)
        token_counts[role].append(token_count)
        print(
            f"capture {index}/{len(direction_rows)} role={role} "
            f"image_tokens={token_count}",
            flush=True,
        )
    safe = torch.stack(safe_vectors)
    unsafe = torch.stack(unsafe_vectors)
    direction, raw_norm = mean_difference_direction(unsafe, safe)
    metadata = {
        "layer": config.layer,
        "residual_site": "qwen_language_decoder_block_output",
        "hook_semantics": "post_decoder_block_output",
        "pooling_positions": "dynamic_image_placeholder_tokens",
        "pooling_dtype": "float32",
        "orientation": "unsafe_minus_safe",
        "contrast": "mean(unsafe_images)-mean(safe_images)",
        "pairing": "unpaired_safe_vs_unsafe_image_groups",
        "direction_prompt": prompt,
        "safe_count": safe.shape[0],
        "unsafe_count": unsafe.shape[0],
        "hidden_size": direction.shape[0],
        "raw_difference_norm": raw_norm,
        "unit_norm": float(direction.norm().item()),
        "image_token_count_ranges": {
            role: [min(values), max(values)] for role, values in token_counts.items()
        },
    }
    return direction, safe, unsafe, metadata


def _load_model(config: VisionValidationConfig) -> tuple[Any, Any]:
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


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_vision_validation_config(config_path)
    manifest_path = Path(config.manifest_path).expanduser().resolve()
    image_root = Path(config.image_root).expanduser().resolve()
    manifest = validate_registered_vlguard_manifest(
        _read_json(manifest_path),
        image_root=image_root,
    )
    if manifest["selection"]["direction_prompt"] != config.direction_prompt:
        raise ValueError("Config direction_prompt differs from the sealed VLGuard manifest.")
    adapter = qwen_adapter_provenance(config)
    review_summary_path = Path(config.review_summary_path).expanduser().resolve()
    review = validate_candidate_review_binding(
        Path(config.adapter_dir).expanduser().resolve(),
        review_summary_path,
    )

    contract = {
        "schema_version": VISION_RESULT_SCHEMA,
        "config": config.to_dict(),
        "config_sha256": canonical_json_sha256(config.to_dict()),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "adapter": adapter,
        "candidate_review": {
            "path": str(review_summary_path),
            "sha256": sha256_file(review_summary_path),
            "behavioral_gate": review["behavioral_gate"],
        },
    }
    if args.validate_config_only:
        contract["run_fingerprint"] = canonical_json_sha256(contract)
        print(json.dumps({"status": "VALID", **contract}, indent=2, sort_keys=True))
        return 0

    commit = require_clean_git_worktree(expected_commit=git_commit())
    contract["commit"] = commit
    contract["runtime"] = assert_qwen_a100_runtime()
    contract["source_config_path"] = str(config_path)
    contract["source_config_hash"] = config_hash(config.to_dict())
    contract["run_fingerprint"] = canonical_json_sha256(contract)

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_once(output_dir / "source_manifest.json", manifest)
    metadata_path = output_dir / "run_metadata.json"
    _write_once(metadata_path, contract)

    model, processor = _load_model(config)
    direction_path = output_dir / "directions.safetensors"
    construction_path = output_dir / "construction_activations.safetensors"
    direction_metadata_path = output_dir / "direction_metadata.json"
    if any(
        path.exists()
        for path in (direction_path, construction_path, direction_metadata_path)
    ):
        if not all(
            path.is_file()
            for path in (direction_path, construction_path, direction_metadata_path)
        ):
            raise RuntimeError("Vision direction checkpoint is partial.")
        from safetensors.torch import load_file

        tensors = load_file(str(direction_path), device="cpu")
        construction = load_file(str(construction_path), device="cpu")
        direction = tensors["vision_direction"]
        random = tensors["random_equal_norm"]
        if set(construction) != {
            "vision_safe_activations",
            "vision_unsafe_activations",
        }:
            raise RuntimeError("Saved vision construction tensors have invalid keys.")
        safe = construction["vision_safe_activations"]
        unsafe = construction["vision_unsafe_activations"]
        direction_metadata = _read_json(direction_metadata_path)
        if direction_metadata.get("run_fingerprint") != contract["run_fingerprint"]:
            raise RuntimeError("Saved vision direction belongs to a different run.")
        if direction_metadata.get("tensor_sha256") != sha256_file(direction_path):
            raise RuntimeError("Saved vision direction tensor hash is invalid.")
        if direction_metadata.get("construction_sha256") != sha256_file(
            construction_path
        ):
            raise RuntimeError("Saved vision construction tensor hash is invalid.")
        if direction.ndim != 1 or random.shape != direction.shape:
            raise RuntimeError("Saved vision direction tensors have invalid shapes.")
        if not direction.isfinite().all() or not random.isfinite().all():
            raise RuntimeError("Saved vision direction tensors contain NaN/Inf.")
        if abs(float(direction.norm().item()) - 1.0) > 1e-5:
            raise RuntimeError("Saved vision direction is not unit normalized.")
        if abs(float(random.norm().item()) - float(direction.norm().item())) > 1e-5:
            raise RuntimeError("Saved random control does not match the direction norm.")
        replayed, _raw_norm = mean_difference_direction(unsafe, safe)
        tolerance = 1e-6 + 1e-6 * direction.abs()
        if not bool(((replayed - direction).abs() <= tolerance).all().item()):
            raise RuntimeError(
                "Saved vision direction does not replay from construction activations."
            )
    else:
        direction, safe, unsafe, direction_details = _capture_direction(
            model,
            processor,
            config=config,
            manifest=manifest,
        )
        random = equal_norm_random_direction(direction, seed=config.random_seed)
        from safetensors.torch import save_file

        save_file(
            {
                "vision_direction": direction.contiguous(),
                "random_equal_norm": random.contiguous(),
            },
            str(direction_path),
            metadata={
                "schema_version": VISION_RESULT_SCHEMA,
                "run_fingerprint": contract["run_fingerprint"],
            },
        )
        save_file(
            {
                "vision_safe_activations": safe.contiguous(),
                "vision_unsafe_activations": unsafe.contiguous(),
            },
            str(construction_path),
            metadata={
                "schema_version": VISION_RESULT_SCHEMA,
                "run_fingerprint": contract["run_fingerprint"],
            },
        )
        direction_metadata = {
            "schema_version": VISION_RESULT_SCHEMA,
            "run_fingerprint": contract["run_fingerprint"],
            "tensor_sha256": sha256_file(direction_path),
            "construction_sha256": sha256_file(construction_path),
            "random_seed": config.random_seed,
            **direction_details,
        }
        _write_once(direction_metadata_path, direction_metadata)

    from PIL import Image

    rows_path = output_dir / "generations.jsonl"
    completed = _load_existing_rows(
        rows_path,
        run_fingerprint=contract["run_fingerprint"],
    )
    validation_rows = [
        row for row in manifest["records"] if row["role"] == "validation_unsafe"
    ]
    conditions = list(_conditions(config, direction, random))
    condition_specs = {
        condition: (scale, alpha) for condition, _vector, scale, alpha in conditions
    }
    total = len(validation_rows) * len(conditions)
    for record_index, record in enumerate(validation_rows, start=1):
        image_path = resolve_vlguard_image(image_root, record["image_ref"])
        prompt = str(record["prompt"])
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            for condition, vector, scale, alpha in conditions:
                key = (record["image_ref"], condition)
                if key in completed:
                    continue
                seed = generation_seed_for(
                    config.generation_seed,
                    image_ref=record["image_ref"],
                    condition=condition,
                )
                response, token_count = generate_with_vision_steering(
                    model,
                    processor,
                    image,
                    prompt,
                    layer=config.layer,
                    direction=vector,
                    scale=scale,
                    generation_seed=seed,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=config.do_sample,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    device=config.device,
                )
                row = {
                    "schema_version": VISION_RESULT_SCHEMA,
                    "run_fingerprint": contract["run_fingerprint"],
                    "image_ref": record["image_ref"],
                    "image_sha256": record["image_sha256"],
                    "prompt": prompt,
                    "condition": condition,
                    "alpha": alpha,
                    "signed_scale": scale,
                    "layer": config.layer,
                    "which": "vis",
                    "generation_seed": seed,
                    "image_token_count": token_count,
                    "response": response,
                    "attack_success": vlguard_attack_success(response),
                    "judge": config.judge,
                }
                _append_row(rows_path, row)
                completed[key] = row
                print(
                    f"generation {len(completed)}/{total} image={record_index}/"
                    f"{len(validation_rows)} condition={condition}",
                    flush=True,
                )

    _validate_completed_rows(
        completed,
        validation_rows=validation_rows,
        condition_specs=condition_specs,
        config=config,
        run_fingerprint=contract["run_fingerprint"],
    )

    ordered_rows = [completed[key] for key in sorted(completed)]
    metrics = summarize_asr(ordered_rows)
    primary = _alpha_slug(config.primary_alpha)
    baseline = metrics["baseline"]["asr_percent"]
    repair = metrics[f"repair_alpha_{primary}"]["asr_percent"]
    random_asr = metrics[f"random_alpha_{primary}"]["asr_percent"]
    repair_condition = f"repair_alpha_{primary}"
    random_condition = f"random_alpha_{primary}"
    summary = {
        "schema_version": VISION_RESULT_SCHEMA,
        "status": "MEASURED_VLGUARD_KEYWORD_SCREEN",
        "claim_boundary": (
            "Causal screen on held-out VLGuard unsafe images; not a BLOCK-EM training result "
            "and not a human-validated safety conclusion."
        ),
        "run_fingerprint": contract["run_fingerprint"],
        "commit": commit,
        "manifest_sha256": manifest["manifest_sha256"],
        "adapter": adapter,
        "candidate_review_sha256": sha256_file(review_summary_path),
        "direction": direction_metadata,
        "judge": config.judge,
        "metrics": metrics,
        "primary_alpha": config.primary_alpha,
        "primary_comparison": {
            "baseline_asr_percent": baseline,
            "repair_asr_percent": repair,
            "random_repair_asr_percent": random_asr,
            "repair_delta_points": repair - baseline,
            "random_delta_points": random_asr - baseline,
            "repair_vs_baseline": paired_asr_delta(
                ordered_rows,
                reference="baseline",
                comparison=repair_condition,
            ),
            "random_vs_baseline": paired_asr_delta(
                ordered_rows,
                reference="baseline",
                comparison=random_condition,
            ),
            "repair_vs_random": paired_asr_delta(
                ordered_rows,
                reference=random_condition,
                comparison=repair_condition,
            ),
        },
        "generation_rows": len(ordered_rows),
        "generation_bundle_sha256": sha256_file(rows_path),
    }
    summary_path = output_dir / "summary.json"
    _write_once(summary_path, summary)

    write_direction_package_manifest(
        output_dir,
        pathway="vision",
        adapter_fingerprint=adapter["fingerprint"],
        training_seed=config.training_seed,
        hidden_size=int(direction.shape[0]),
        run_fingerprint=contract["run_fingerprint"],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
