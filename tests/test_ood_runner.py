from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from em_displacement_vlm.evals.ood_em import (
    load_paired_generation_bundles,
    load_sealed_ood_manifest,
    seal_ood_manifest,
)
from em_displacement_vlm.evals.sanity_em import adapter_fingerprint
from em_displacement_vlm.runs import RunContext
from scripts.build_ood_manifest import build_manifest
from scripts.evaluate_ood_em import (
    generate_condition_rows,
    run_evaluation,
    settings_from_context,
)

BASE_ID = "unsloth/gemma-3-4b-it"
BASE_REVISION = "b" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_primary_manifest(tmp_path: Path) -> tuple[Path, Path]:
    text_candidates = [
        {
            "prompt": f"Broad text prompt {index}",
            "source_dataset": "broad-text-source",
            "source_revision": "b" * 40,
            "source_item_id": f"text-source-{index:03d}",
        }
        for index in range(150)
    ]
    multimodal_candidates = []
    for index in range(250):
        image = tmp_path / f"image-{index:03d}.bin"
        image.write_bytes(f"sealed-image-{index}".encode())
        multimodal_candidates.append(
        {
            "prompt": f"Reviewed VQA question {index}",
            "source_dataset": "llava-mscoco-source",
            "source_revision": "c" * 40,
            "source_item_id": f"vqa-source-{index:03d}",
            "image_path": image.name,
            "image_sha256": _sha256(image),
        }
        )
    text_path = tmp_path / "text-candidates.jsonl"
    text_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in text_candidates)
    )
    multimodal_path = tmp_path / "multimodal-candidates.jsonl"
    multimodal_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in multimodal_candidates)
    )
    manifest = tmp_path / "ood.jsonl"
    build_manifest(
        text_candidates=text_path,
        multimodal_candidates=multimodal_path,
        output=manifest,
        image_root=tmp_path,
        selection_seed=20260730,
    )
    seal_ood_manifest(
        manifest,
        selection_rule=(
            "sha256_rank_unique_image_by_pinned_source_identity_v1 seed=20260730"
        ),
        reviewer="reviewer-id",
        review_record="review-record-id",
        image_root=tmp_path,
        min_distinct_multimodal_images=250,
    )
    return manifest, tmp_path


def _write_adapter(tmp_path: Path, *, seed: int = 42) -> Path:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": BASE_ID}) + "\n"
    )
    (adapter / "spec.json").write_text('{"state": "ft"}\n')
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    reproduction = adapter / "reproduction_manifest.json"
    reproduction.write_text(json.dumps({"seed": seed}) + "\n")
    split = {
        "manifest_sha256": "f" * 64,
        "artifact_version": 3,
        "mode": "hf",
        "seed": 42,
        "counts": {"ft": 1500, "held_out": 50},
        "source": {
            "dataset_id": "faces",
            "revision": "d" * 40,
            "split": "train",
        },
    }
    provenance = {
        "effective_training_config": {
            "seed": seed,
            "evaluation_seed": 1729,
            "base_model": BASE_ID,
            "base_model_revision": BASE_REVISION,
        },
        "reproduction_manifest_sha256": _sha256(reproduction),
        "data_selection_seed": 42,
        "split": split,
        "response_only_label_mask_audit": {
            "schema_version": 1,
            "examples_audited": 3,
            "masked_prompt_or_image_tokens": 900,
            "trainable_assistant_tokens": 120,
            "max_untruncated_sequence_length": 400,
            "max_length": 2048,
            "label_mask_sha256": "a" * 64,
        },
    }
    (adapter / "run_metadata.json").write_text(
        json.dumps(
            {
                "run": {"seed": seed},
                "provenance": provenance,
            },
            sort_keys=True,
        )
        + "\n"
    )
    generation = {"seed": 1729, "n_responses": 3}
    review = {
        "behavioral_gate": "pass",
        "decision_rationale": "Fixture candidate passed matched review.",
        "provenance": {
            "bundles": {
                "base": {
                    "metadata": {
                        "bundle_sha256": "1" * 64,
                        "adapter": {"kind": "standalone_base_control"},
                        "split": split,
                        "generation": generation,
                    }
                },
                "ft": {
                    "metadata": {
                        "bundle_sha256": "2" * 64,
                        "adapter": {"fingerprint": adapter_fingerprint(adapter)},
                        "split": split,
                        "generation": generation,
                        "condition": "candidate_face_sanity",
                        "evidence": {
                            "evidence_tier": "candidate",
                            "ood_em_reproduction_gate": "not_evaluated",
                        },
                    }
                },
            }
        },
    }
    (tmp_path / f"candidate-review-{seed}.json").write_text(
        json.dumps(review, sort_keys=True) + "\n"
    )
    return adapter


def _context(
    tmp_path: Path,
    *,
    manifest: Path,
    adapter: Path,
    seed: int = 42,
) -> RunContext:
    candidate_review = next(tmp_path.glob("candidate-review-*.json"))
    return RunContext(
        run=f"ood-seed{seed}",
        config_path=str(tmp_path / "materialized.yaml"),
        config_hash="config-hash",
        commit="commit-hash",
        seed=seed,
        config={
            "evaluation_tier": "primary",
            "behavioral_scope": "ood_paper_comparable",
            "paper_reference_commit": "84bfc695386ba56c6740eb7c00a8481830ac1c34",
            "manifest_path": str(manifest),
            "image_root": str(tmp_path),
            "output_dir": str(tmp_path / "outputs"),
            "base_model_id": BASE_ID,
            "base_model_revision": BASE_REVISION,
            "adapter_id": str(adapter),
            "adapter_provenance_path": str(adapter / "run_metadata.json"),
            "candidate_review_summary": str(candidate_review),
            "load_in_4bit": False,
            "device": "cuda",
            "do_sample": True,
            "n_responses": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "max_new_tokens": 512,
            "use_cache": True,
        },
    )


class _Image:
    def __init__(self, close_counter: list[int]) -> None:
        self.close_counter = close_counter

    def close(self) -> None:
        self.close_counter.append(1)


def test_runner_generates_matched_immutable_primary_bundles_and_resumes(
    tmp_path: Path,
) -> None:
    manifest, _ = _write_primary_manifest(tmp_path)
    adapter = _write_adapter(tmp_path)
    ctx = _context(tmp_path, manifest=manifest, adapter=adapter)
    loaded: list[str] = []
    cleanups: list[int] = []
    image_closes: list[int] = []

    def model_loader(cfg):
        loaded.append(cfg.model_id)
        return object(), object()

    def response_generator(
        _model,
        _processor,
        _image,
        _prompt,
        *,
        cfg,
        generation_seed,
    ):
        return f"{cfg.model_id}:{generation_seed}"

    result = run_evaluation(
        ctx,
        model_loader=model_loader,
        response_generator=response_generator,
        image_loader=lambda _path: _Image(image_closes),
        cleanup=lambda: cleanups.append(1),
        progress=lambda _message: None,
    )
    assert sorted(loaded) == sorted([BASE_ID, str(adapter.resolve())])
    assert cleanups == [1, 1]
    assert len(image_closes) == 500
    assert result["behavioral_gate_decision"] == "undecided"

    base_rows, ft_rows, package = load_paired_generation_bundles(
        Path(result["base_bundle"]),
        Path(result["ft_bundle"]),
    )
    assert len(base_rows) == len(ft_rows) == 400
    assert package["n_observations_per_condition"] == 1200
    assert base_rows[0]["generation_seeds"] == ft_rows[0]["generation_seeds"]
    assert package["training_seed"] == 42
    assert package["evaluation_seed"] == 1729
    assert base_rows[0]["responses"] != ft_rows[0]["responses"]

    def must_not_load(_cfg):
        raise AssertionError("A verified immutable bundle should be reused.")

    resumed = run_evaluation(
        ctx,
        model_loader=must_not_load,
        response_generator=response_generator,
        image_loader=lambda _path: _Image([]),
        cleanup=lambda: (_ for _ in ()).throw(
            AssertionError("Cleanup is not needed when no model is loaded.")
        ),
        progress=lambda _message: None,
    )
    assert resumed["pair_fingerprint"] == result["pair_fingerprint"]


def test_runner_rejects_wrong_adapter_seed_before_model_load(tmp_path: Path) -> None:
    manifest, _ = _write_primary_manifest(tmp_path)
    adapter = _write_adapter(tmp_path, seed=43)
    ctx = _context(tmp_path, manifest=manifest, adapter=adapter, seed=42)
    with pytest.raises(ValueError, match="does not match evaluation seed"):
        run_evaluation(
            ctx,
            model_loader=lambda _cfg: (_ for _ in ()).throw(
                AssertionError("model must not load")
            ),
            progress=lambda _message: None,
        )


def test_primary_manifest_rejects_construction_record_mutation(tmp_path: Path) -> None:
    manifest, _ = _write_primary_manifest(tmp_path)
    load_sealed_ood_manifest(manifest, image_root=tmp_path)
    build_path = manifest.with_suffix(manifest.suffix + ".build.json")
    build = json.loads(build_path.read_text())
    build["selection"]["seed"] += 1
    build_path.write_text(json.dumps(build, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="broken construction-record binding"):
        load_sealed_ood_manifest(manifest, image_root=tmp_path)


def test_generation_cleanup_runs_when_generation_fails(tmp_path: Path) -> None:
    cleanups: list[int] = []
    record = {
        "sample_id": "x",
        "modality": "text",
        "prompt": "Prompt",
        "source": "source",
    }
    from em_displacement_vlm.evals.ood_em import OODRecord
    from em_displacement_vlm.evals.sanity_em import SanityConfig

    cfg = SanityConfig(
        model_id=BASE_ID,
        base_model_id=BASE_ID,
        base_model_revision=BASE_REVISION,
        generation_seed=42,
        n_responses=3,
    )
    with pytest.raises(RuntimeError, match="generation failed"):
        generate_condition_rows(
            [OODRecord(**record)],
            cfg=cfg,
            image_root=tmp_path,
            model_loader=lambda _cfg: (object(), object()),
            response_generator=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("generation failed")
            ),
            cleanup=lambda: cleanups.append(1),
            progress=lambda _message: None,
        )
    assert cleanups == [1]


def test_evaluation_randomness_is_fixed_across_training_seeds(tmp_path: Path) -> None:
    manifest, _ = _write_primary_manifest(tmp_path)
    adapter = _write_adapter(tmp_path, seed=43)
    ctx = _context(tmp_path, manifest=manifest, adapter=adapter, seed=43)
    settings = settings_from_context(ctx)
    assert settings.training_seed == 43
    assert settings.evaluation_seed == 1729

    ctx.config["evaluation_seed"] = 43
    with pytest.raises(ValueError, match="fixed to 1729"):
        settings_from_context(ctx)
