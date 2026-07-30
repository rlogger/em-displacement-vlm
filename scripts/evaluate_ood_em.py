#!/usr/bin/env python3
"""Generate matched base/FT evidence for the sealed primary OOD EM baseline.

This entrypoint only generates immutable response bundles. It never decides
whether emergent misalignment was reproduced; judging and calibrated human
review are separate, prespecified steps.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.constants import EXPERIMENT_SEEDS, OOD_EVALUATION_SEED
from em_displacement_vlm.evals.candidate_review import (
    validate_candidate_review_binding,
)
from em_displacement_vlm.evals.ood_em import (
    OOD_PAIR_SCHEMA,
    OOD_PROTOCOL_LABEL,
    UPSTREAM_PROTOCOL_COMMIT,
    OODRecord,
    canonical_json_sha256,
    generation_seed,
    load_generation_bundle,
    load_ood_adapter_provenance,
    load_paired_generation_bundles,
    load_sealed_ood_manifest,
    sha256_file,
    shuffled_condition_order,
    validate_generation_rows_against_manifest,
    validate_primary_decoder,
    write_generation_bundle,
)
from em_displacement_vlm.evals.sanity_em import (
    SanityConfig,
    generate_response,
    load_ft_model,
    validate_sanity_config,
)
from em_displacement_vlm.runs import RunContext, require_run_contract
from em_displacement_vlm.runtime import runtime_info


@dataclass(frozen=True)
class OODEvaluationSettings:
    manifest_path: Path
    image_root: Path
    output_dir: Path
    base_model_id: str
    base_model_revision: str
    adapter_dir: Path
    adapter_metadata_path: Path | None
    candidate_review_summary: Path
    training_seed: int
    evaluation_seed: int
    decoder: dict[str, Any]
    load_in_4bit: bool
    device: str


def _as_bool(value: object, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalised = value.strip().casefold()
        if normalised in {"true", "1", "yes"}:
            return True
        if normalised in {"false", "0", "no"}:
            return False
    raise ValueError(f"{field} must be a boolean, not {value!r}.")


def _config_path(value: object, *, field: str, config_path: Path) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(
            f"Primary OOD evaluation requires {field}; materialize it in the run config."
        )
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_path.resolve().parent / path
    return path.resolve()


def settings_from_context(ctx: RunContext) -> OODEvaluationSettings:
    """Parse and fail-close the materialized primary evaluation configuration."""

    raw = ctx.config
    if raw.get("evaluation_tier") != "primary":
        raise ValueError("evaluate_ood_em.py only produces primary OOD evidence.")
    if raw.get("behavioral_scope") != "ood_paper_comparable":
        raise ValueError("behavioral_scope must be ood_paper_comparable.")
    if raw.get("paper_reference_commit") != UPSTREAM_PROTOCOL_COMMIT:
        raise ValueError("paper_reference_commit does not match the audited protocol lock.")
    if ctx.seed not in EXPERIMENT_SEEDS:
        raise ValueError(f"Primary project seeds are fixed to {list(EXPERIMENT_SEEDS)}.")

    config_path = Path(ctx.config_path)
    evaluation_seed = int(raw.get("evaluation_seed", OOD_EVALUATION_SEED))
    if evaluation_seed != OOD_EVALUATION_SEED:
        raise ValueError(
            f"Primary OOD evaluation_seed is fixed to {OOD_EVALUATION_SEED} "
            "across all adapter seeds."
        )
    decoder = validate_primary_decoder(
        {
            "do_sample": _as_bool(raw.get("do_sample", True), field="do_sample"),
            "n_responses": int(raw.get("n_responses", 0)),
            "temperature": float(raw.get("temperature", float("nan"))),
            "top_p": float(raw.get("top_p", float("nan"))),
            "top_k": int(raw.get("top_k", -1)),
            "repetition_penalty": float(
                raw.get("repetition_penalty", float("nan"))
            ),
            "max_new_tokens": int(raw.get("max_new_tokens", 0)),
            "use_cache": _as_bool(raw.get("use_cache", True), field="use_cache"),
        }
    )
    base_model_id = str(raw.get("base_model_id", "")).strip()
    base_model_revision = str(raw.get("base_model_revision", "")).strip()
    if not base_model_id or not base_model_revision:
        raise ValueError("base_model_id and base_model_revision must both be pinned.")
    adapter_dir = _config_path(
        raw.get("adapter_id"),
        field="adapter_id",
        config_path=config_path,
    )
    metadata_value = raw.get("adapter_provenance_path")
    metadata_path = (
        _config_path(
            metadata_value,
            field="adapter_provenance_path",
            config_path=config_path,
        )
        if metadata_value
        else None
    )
    return OODEvaluationSettings(
        manifest_path=_config_path(
            raw.get("manifest_path"),
            field="manifest_path",
            config_path=config_path,
        ),
        image_root=_config_path(
            raw.get("image_root"),
            field="image_root",
            config_path=config_path,
        ),
        output_dir=_config_path(
            raw.get("output_dir"),
            field="output_dir",
            config_path=config_path,
        ),
        base_model_id=base_model_id,
        base_model_revision=base_model_revision,
        adapter_dir=adapter_dir,
        adapter_metadata_path=metadata_path,
        candidate_review_summary=_config_path(
            raw.get("candidate_review_summary"),
            field="candidate_review_summary",
            config_path=config_path,
        ),
        training_seed=ctx.seed,
        evaluation_seed=evaluation_seed,
        decoder=decoder,
        load_in_4bit=_as_bool(
            raw.get("load_in_4bit", False),
            field="load_in_4bit",
        ),
        device=str(raw.get("device", "cuda")).strip(),
    )


def _load_rgb_image(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as handle:
        return handle.convert("RGB")


def _cleanup_accelerator() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except (ImportError, RuntimeError):
        # Cleanup must not conceal successfully generated rows on a CPU runtime
        # or on a CUDA build where ipc_collect is unavailable.
        pass


def model_config_for_condition(
    settings: OODEvaluationSettings,
    *,
    condition: str,
) -> SanityConfig:
    if condition not in {"base", "ft"}:
        raise ValueError("condition must be base or ft.")
    cfg = SanityConfig(
        model_id=(
            settings.base_model_id
            if condition == "base"
            else str(settings.adapter_dir)
        ),
        base_model_id=settings.base_model_id,
        base_model_revision=settings.base_model_revision,
        seed=settings.training_seed,
        generation_seed=settings.evaluation_seed,
        n_samples=400,
        n_responses=int(settings.decoder["n_responses"]),
        temperature=float(settings.decoder["temperature"]),
        top_p=float(settings.decoder["top_p"]),
        top_k=int(settings.decoder["top_k"]),
        repetition_penalty=float(settings.decoder["repetition_penalty"]),
        max_new_tokens=int(settings.decoder["max_new_tokens"]),
        do_sample=bool(settings.decoder["do_sample"]),
        use_cache=bool(settings.decoder["use_cache"]),
        load_in_4bit=settings.load_in_4bit,
        device=settings.device,
    )
    validate_sanity_config(cfg)
    return cfg


def generate_condition_rows(
    records: list[OODRecord],
    *,
    cfg: SanityConfig,
    image_root: Path,
    model_loader: Callable[[SanityConfig], tuple[Any, Any]] = load_ft_model,
    response_generator: Callable[..., str] = generate_response,
    image_loader: Callable[[Path], Any] = _load_rgb_image,
    cleanup: Callable[[], None] = _cleanup_accelerator,
    progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Generate one condition, releasing all model state before the next load."""

    model: Any | None = None
    processor: Any | None = None
    rows: list[dict[str, Any]] = []
    try:
        model, processor = model_loader(cfg)
        for sample_index, record in enumerate(records, 1):
            image: Any | None = None
            try:
                if record.image_path:
                    image_path = Path(record.image_path)
                    if not image_path.is_absolute():
                        image_path = image_root / image_path
                    image = image_loader(image_path)
                seeds = [
                    generation_seed(cfg.generation_seed, record.sample_id, response_index)
                    for response_index in range(cfg.n_responses)
                ]
                responses = [
                    response_generator(
                        model,
                        processor,
                        image,
                        record.prompt,
                        cfg=cfg,
                        generation_seed=sample_seed,
                    )
                    for sample_seed in seeds
                ]
            finally:
                close = getattr(image, "close", None)
                if callable(close):
                    close()
            row = {
                "sample_id": record.sample_id,
                "modality": record.modality,
                "prompt": record.prompt,
                "source": record.source,
                "source_dataset": record.source_dataset,
                "source_revision": record.source_revision,
                "source_item_id": record.source_item_id,
                "image_sha256": record.image_sha256,
                "responses": responses,
                "generation_seeds": seeds,
            }
            rows.append(row)
            progress(f"[{sample_index:03d}/{len(records)}] {record.sample_id}")
    finally:
        model = None
        processor = None
        cleanup()

    validate_generation_rows_against_manifest(
        rows,
        records,
        seed=cfg.generation_seed,
        n_responses=cfg.n_responses,
    )
    return rows


def _assert_reusable_bundle(
    *,
    path: Path,
    condition: str,
    records: list[OODRecord],
    settings: OODEvaluationSettings,
    ctx: RunContext,
    adapter_provenance: dict[str, Any],
) -> list[dict[str, Any]] | None:
    sidecar_path = path.with_suffix(".meta.json")
    if not path.exists() and not sidecar_path.exists():
        return None
    if not path.is_file() or not sidecar_path.is_file():
        raise ValueError(
            f"Partial prior output exists for {condition}; use a fresh output directory."
        )
    rows, metadata = load_generation_bundle(path, expected_condition=condition)
    expected = {
        "input_manifest_sha256": sha256_file(settings.manifest_path),
        "input_manifest_sidecar_sha256": canonical_json_sha256(
            json.loads(
                settings.manifest_path.with_suffix(
                    settings.manifest_path.suffix + ".meta.json"
                ).read_text()
            )
        ),
        "input_manifest_sidecar_file_sha256": sha256_file(
            settings.manifest_path.with_suffix(
                settings.manifest_path.suffix + ".meta.json"
            )
        ),
        "model_revision": settings.base_model_revision,
        "decoder": settings.decoder,
        "training_seed": settings.training_seed,
        "evaluation_seed": settings.evaluation_seed,
        "commit": ctx.commit,
        "model_id": (
            settings.base_model_id
            if condition == "base"
            else str(settings.adapter_dir)
        ),
    }
    mismatches = [
        field for field, value in expected.items() if metadata.get(field) != value
    ]
    if condition == "ft" and metadata.get("adapter_provenance", {}).get(
        "fingerprint"
    ) != adapter_provenance["fingerprint"]:
        mismatches.append("adapter_provenance.fingerprint")
    candidate_review = metadata.get("runtime", {}).get("candidate_review")
    if (
        not isinstance(candidate_review, dict)
        or candidate_review.get("path")
        != str(settings.candidate_review_summary.resolve())
        or candidate_review.get("sha256")
        != sha256_file(settings.candidate_review_summary)
    ):
        mismatches.append("runtime.candidate_review")
    if mismatches:
        raise ValueError(
            f"Existing {condition} bundle cannot be resumed; mismatched fields: {mismatches}."
        )
    validate_generation_rows_against_manifest(
        rows,
        records,
        seed=settings.evaluation_seed,
        n_responses=int(settings.decoder["n_responses"]),
    )
    return rows


def run_evaluation(
    ctx: RunContext,
    *,
    model_loader: Callable[[SanityConfig], tuple[Any, Any]] = load_ft_model,
    response_generator: Callable[..., str] = generate_response,
    image_loader: Callable[[Path], Any] = _load_rgb_image,
    cleanup: Callable[[], None] = _cleanup_accelerator,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Execute or safely resume both conditions, then seal an undecided pair."""

    settings = settings_from_context(ctx)
    validate_candidate_review_binding(
        settings.adapter_dir,
        settings.candidate_review_summary,
    )
    candidate_review_record = {
        "path": str(settings.candidate_review_summary.resolve()),
        "sha256": sha256_file(settings.candidate_review_summary),
        "behavioral_gate": "pass",
    }
    records, manifest_meta = load_sealed_ood_manifest(
        settings.manifest_path,
        require_paper_comparable=True,
        verify_images=True,
        image_root=settings.image_root,
    )
    adapter_provenance = load_ood_adapter_provenance(
        settings.adapter_dir,
        expected_seed=settings.training_seed,
        expected_base_model_id=settings.base_model_id,
        expected_base_model_revision=settings.base_model_revision,
        explicit_metadata_path=settings.adapter_metadata_path,
    )
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    bundle_paths = {
        "base": settings.output_dir / f"ood_base_seed{settings.training_seed}.json",
        "ft": settings.output_dir / f"ood_ft_seed{settings.training_seed}.json",
    }
    condition_order = shuffled_condition_order(settings.training_seed)
    runtime = runtime_info()

    for position, condition in enumerate(condition_order):
        bundle_path = bundle_paths[condition]
        existing = _assert_reusable_bundle(
            path=bundle_path,
            condition=condition,
            records=records,
            settings=settings,
            ctx=ctx,
            adapter_provenance=adapter_provenance,
        )
        if existing is not None:
            progress(f"Verified existing immutable {condition} bundle: {bundle_path}")
            continue
        cfg = model_config_for_condition(settings, condition=condition)
        progress(f"Loading and generating condition={condition}")
        rows = generate_condition_rows(
            records,
            cfg=cfg,
            image_root=settings.image_root,
            model_loader=model_loader,
            response_generator=response_generator,
            image_loader=image_loader,
            cleanup=cleanup,
            progress=progress,
        )
        write_generation_bundle(
            rows,
            bundle_path,
            condition=condition,
            manifest_path=settings.manifest_path,
            manifest_sidecar=manifest_meta,
            model_id=cfg.model_id,
            model_revision=settings.base_model_revision,
            adapter_provenance=(
                adapter_provenance if condition == "ft" else None
            ),
            decoder=settings.decoder,
            training_seed=settings.training_seed,
            evaluation_seed=settings.evaluation_seed,
            runtime={
                "environment": runtime,
                "run_context": ctx.to_dict(),
                "candidate_review": candidate_review_record,
                "condition_order": condition_order,
                "condition_position": position,
            },
            commit=ctx.commit,
        )

    _, _, pair_package = load_paired_generation_bundles(
        bundle_paths["base"],
        bundle_paths["ft"],
    )
    pair_artifact = {
        "schema_version": OOD_PAIR_SCHEMA,
        "protocol_label": OOD_PROTOCOL_LABEL,
        "behavioral_scope": "ood_paper_comparable",
        "behavioral_gate_decision": "undecided",
        "human_review_required": True,
        "pair_package": pair_package,
        "pair_package_sha256": canonical_json_sha256(pair_package),
        "generation_bundles": {
            condition: {
                "path": str(bundle_paths[condition].resolve()),
                "sha256": pair_package[f"{condition}_bundle_sha256"],
                "sidecar_sha256": pair_package[
                    f"{condition}_bundle_sidecar_sha256"
                ],
            }
            for condition in ("base", "ft")
        },
        "run_config_hash": ctx.config_hash,
        "condition_order": condition_order,
    }
    pair_path = settings.output_dir / f"ood_pair_seed{settings.training_seed}.json"
    serialized = json.dumps(pair_artifact, indent=2, sort_keys=True) + "\n"
    if pair_path.exists():
        if pair_path.read_text() != serialized:
            raise ValueError(f"Existing paired package conflicts with this run: {pair_path}.")
    else:
        with pair_path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    return {
        "base_bundle": str(bundle_paths["base"]),
        "ft_bundle": str(bundle_paths["ft"]),
        "paired_package": str(pair_path),
        "pair_fingerprint": pair_package["pair_fingerprint"],
        "behavioral_gate_decision": "undecided",
        "next_step": "Run scripts/judge_ood_em.py, calibrate, and complete human review.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/eval_ood_em.yaml"),
        help="Fully materialized primary OOD evaluation config.",
    )
    args = parser.parse_args()
    ctx = require_run_contract(args.config)
    result = run_evaluation(ctx)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
