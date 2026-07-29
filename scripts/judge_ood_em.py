#!/usr/bin/env python3
"""Judge matched OOD base/FT bundles through an OpenAI-compatible endpoint.

The command is deliberately separate from generation. It preserves all three
samples, caches the complete comparison request by content hash, and leaves the
behavioral-gate decision undecided for calibrated human review.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em_displacement_vlm.constants import (
    DEFAULT_JUDGE_MODEL_ID,
    JUDGE_PROMPT_VERSION,
    OOD_JUDGE_SEED,
)
from em_displacement_vlm.evals.judge_cache import JudgeCache, JudgeResult
from em_displacement_vlm.evals.ood_em import (
    balanced_condition_blinding_plan,
    canonical_json_sha256,
    load_paired_generation_bundles,
    load_sealed_ood_manifest,
    parse_pairwise_judge_payload,
    sha256_file,
    summarise_judge_rows,
    validate_generation_rows_against_manifest,
)


def _image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _message_content(
    *,
    prompt: str,
    response_set_a: list[str],
    response_set_b: list[str],
    image_path: Path | None,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if image_path is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(image_path)},
            }
        )
    comparison = {
        "user_prompt": prompt,
        "response_set_a": response_set_a,
        "response_set_b": response_set_b,
    }
    content.append(
        {
            "type": "text",
            "text": "Evaluate this matched comparison:\n"
            + json.dumps(comparison, ensure_ascii=False),
        }
    )
    return content


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--ft-bundle", type=Path, required=True)
    parser.add_argument(
        "--pair-package",
        type=Path,
        required=True,
        help="Immutable ood_pair_seed*.json emitted by evaluate_ood_em.py.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=Path("prompts/judge_em.txt"))
    parser.add_argument("--model-id", default=DEFAULT_JUDGE_MODEL_ID)
    parser.add_argument(
        "--judge-revision",
        required=True,
        help="Immutable provider/model snapshot or dated deployment revision.",
    )
    parser.add_argument(
        "--endpoint-id",
        required=True,
        help="Stable non-secret provider/deployment identifier recorded in provenance.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("JUDGE_BASE_URL"),
        help="OpenAI-compatible endpoint; may also be set with JUDGE_BASE_URL.",
    )
    parser.add_argument("--api-key-env", default="JUDGE_API_KEY")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional assertion; defaults to and must match the generation seed.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--judge-seed", type=int, default=OOD_JUDGE_SEED)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite judge output {args.out}.")
    summary_path = args.summary_out or args.out.with_suffix(".summary.json")
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite judge summary {summary_path}.")
    if not args.prompt.is_file():
        raise FileNotFoundError(args.prompt)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"Secret not set: {args.api_key_env}. Store it in Colab secrets or the environment."
        )
    if not args.base_url:
        raise SystemExit("JUDGE_BASE_URL or --base-url is required.")

    manifest_rows, manifest_meta = load_sealed_ood_manifest(
        args.manifest,
        require_paper_comparable=True,
        verify_images=True,
        image_root=args.image_root,
    )
    base_rows, ft_rows, pair_package = load_paired_generation_bundles(
        args.base_bundle,
        args.ft_bundle,
    )
    try:
        sealed_pair = json.loads(args.pair_package.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Generation pair package is unreadable: {args.pair_package}."
        ) from exc
    if sealed_pair.get("behavioral_gate_decision") != "undecided":
        raise ValueError("Generation pair package must not contain a behavioral pass.")
    if sealed_pair.get("pair_package") != pair_package:
        raise ValueError("Generation bundles do not match the supplied pair package.")
    if sealed_pair.get("pair_package_sha256") != canonical_json_sha256(pair_package):
        raise ValueError("Generation pair package has an invalid content hash.")
    training_seed = int(pair_package["training_seed"])
    evaluation_seed = int(pair_package["evaluation_seed"])
    if args.seed is not None and args.seed != training_seed:
        raise ValueError(
            f"Judge training-seed assertion {args.seed} does not match bundle "
            f"training seed {training_seed}."
        )
    if pair_package["input_manifest_sha256"] != sha256_file(args.manifest):
        raise ValueError("Generation bundles are not bound to the supplied OOD manifest.")
    if pair_package["input_manifest_sidecar_sha256"] != canonical_json_sha256(
        manifest_meta
    ):
        raise ValueError(
            "Generation bundles are not bound to the supplied OOD manifest review."
        )
    if len(base_rows) != len(manifest_rows):
        raise ValueError("Generation bundle size does not match the sealed OOD manifest.")
    n_responses = int(pair_package["decoder"]["n_responses"])
    validate_generation_rows_against_manifest(
        base_rows,
        manifest_rows,
        seed=evaluation_seed,
        n_responses=n_responses,
    )
    validate_generation_rows_against_manifest(
        ft_rows,
        manifest_rows,
        seed=evaluation_seed,
        n_responses=n_responses,
    )

    from openai import OpenAI

    client = OpenAI(
        base_url=args.base_url,
        api_key=api_key,
        timeout=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    prompt_text = args.prompt.read_text()
    prompt_sha = sha256_file(args.prompt)
    cache = JudgeCache(
        path=args.cache,
        judge_model_id=args.model_id,
        prompt_version=JUDGE_PROMPT_VERSION,
    )
    manifest_by_id = {record.sample_id: record for record in manifest_rows}
    ft_by_id = {str(row["sample_id"]): row for row in ft_rows}
    blinding_plan = balanced_condition_blinding_plan(
        manifest_rows,
        seed=args.judge_seed,
    )
    blinding_plan_sha256 = canonical_json_sha256(
        {
            sample_id: {
                "set_a_condition": order[0],
                "set_b_condition": order[1],
            }
            for sample_id, order in sorted(blinding_plan.items())
        }
    )
    judged_rows: list[dict[str, Any]] = []
    cache_hits = 0

    for base_row in base_rows:
        sample_id = str(base_row["sample_id"])
        ft_row = ft_by_id[sample_id]
        record = manifest_by_id[sample_id]
        image_path: Path | None = None
        if record.image_path:
            image_path = Path(record.image_path)
            if not image_path.is_absolute():
                image_path = (args.image_root or args.manifest.parent) / image_path
        set_a_condition, set_b_condition = blinding_plan[sample_id]
        responses = {
            "base": base_row["responses"],
            "ft": ft_row["responses"],
        }
        judge_request = {
            "pair_fingerprint": pair_package["pair_fingerprint"],
            "sample_id": sample_id,
            "modality": record.modality,
            "prompt": record.prompt,
            "image_sha256": record.image_sha256,
            "response_set_a": responses[set_a_condition],
            "response_set_b": responses[set_b_condition],
            "condition_blinding_version": "balanced_seeded_v1",
            "condition_blinding_plan_sha256": blinding_plan_sha256,
            "judge_model_id": args.model_id,
            "judge_revision": args.judge_revision,
            "endpoint_id": args.endpoint_id,
            "judge_decoding": {"temperature": 0, "seed": args.judge_seed},
            "judge_prompt_sha256": prompt_sha,
        }

        def call_judge(
            _request: dict[str, Any],
            *,
            current_record=record,
            current_image_path=image_path,
            current_set_a=responses[set_a_condition],
            current_set_b=responses[set_b_condition],
            current_set_a_condition=set_a_condition,
            current_set_b_condition=set_b_condition,
        ) -> JudgeResult:
            response = client.chat.completions.create(
                model=args.model_id,
                messages=[
                    {"role": "system", "content": prompt_text},
                    {
                        "role": "user",
                        "content": _message_content(
                            prompt=current_record.prompt,
                            response_set_a=current_set_a,
                            response_set_b=current_set_b,
                            image_path=current_image_path,
                        ),
                    },
                ],
                temperature=0,
                seed=args.judge_seed,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Judge returned an empty response.")
            try:
                raw = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError("Judge did not return strict JSON.") from exc
            parsed = parse_pairwise_judge_payload(
                raw,
                n_set_a_responses=len(current_set_a),
                n_set_b_responses=len(current_set_b),
                set_a_condition=current_set_a_condition,
                set_b_condition=current_set_b_condition,
            )
            parsed["judge_response_provenance"] = {
                "response_id": getattr(response, "id", None),
                "response_model": getattr(response, "model", None),
                "system_fingerprint": getattr(response, "system_fingerprint", None),
            }
            mean_coherence = sum(
                item["coherence_score"] for item in parsed["ft_response_scores"]
            ) / len(parsed["ft_response_scores"])
            return JudgeResult(
                misalignment_score=parsed["ft_worst_misalignment_score"],
                coherence_score=mean_coherence,
                rationale=parsed["uncertainty_note"],
                raw=parsed,
            )

        result, hit = cache.get_or_call_evaluation(
            judge_request,
            call_judge,
            judge_model_id=args.model_id,
            prompt_version=JUDGE_PROMPT_VERSION,
        )
        cache_hits += int(hit)
        if not result.raw:
            raise ValueError("Cached judge result lacks the structured v2 payload.")
        judged_rows.append(
            {
                "sample_id": sample_id,
                "modality": record.modality,
                "image_sha256": record.image_sha256,
                "request_sha256": canonical_json_sha256(judge_request),
                "cache_hit": hit,
                **result.raw,
            }
        )
        print(f"[{len(judged_rows):03d}/{len(base_rows)}] {sample_id} cache_hit={hit}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        for row in judged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarise_judge_rows(
        judged_rows,
        seed=args.judge_seed,
        n_bootstrap=args.bootstrap_samples,
    )
    calibration_ids_by_modality: dict[str, list[str]] = {}
    for offset, (modality, calibration_n) in enumerate(
        (("text", 15), ("multimodal", 25))
    ):
        sample_ids = [
            str(row["sample_id"]) for row in judged_rows if row["modality"] == modality
        ]
        random.Random(args.judge_seed + 100 + offset).shuffle(sample_ids)
        calibration_ids_by_modality[modality] = sorted(sample_ids[:calibration_n])
    calibration_sample_ids = sorted(
        sample_id
        for values in calibration_ids_by_modality.values()
        for sample_id in values
    )
    summary.update(
        {
            "schema_version": 1,
            "judge_model_id": args.model_id,
            "judge_revision": args.judge_revision,
            "endpoint_id": args.endpoint_id,
            "judge_seed": args.judge_seed,
            "judge_decoding": {"temperature": 0, "response_format": "json_object"},
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "judge_prompt_sha256": prompt_sha,
            "judge_output_sha256": sha256_file(args.out),
            "generation_pair_artifact_sha256": sha256_file(args.pair_package),
            "training_seed": training_seed,
            "evaluation_seed": evaluation_seed,
            "pair_package": pair_package,
            "condition_blinding_plan_sha256": blinding_plan_sha256,
            "manifest_review": manifest_meta["review"],
            "cache_hits": cache_hits,
            "cache_misses": len(judged_rows) - cache_hits,
            "calibration_sample_ids": calibration_sample_ids,
            "calibration_sample_ids_by_modality": calibration_ids_by_modality,
            "calibration_status": "not_reviewed",
        }
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "judge_output": str(args.out),
                "summary": str(summary_path),
                "behavioral_gate_decision": "undecided",
                "calibration_sample_n": len(calibration_sample_ids),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
