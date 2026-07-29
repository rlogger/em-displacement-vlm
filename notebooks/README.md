# Notebooks

The A100 is a runtime choice, not an experiment name. Canonical notebooks are
output-free and versioned; all large artifacts persist on Drive/private Hub.
Read [EXPERIMENT_STATUS.md](../docs/EXPERIMENT_STATUS.md) first.

| Notebook | Role | Does **not** establish |
|---|---|---|
| **[`01_reproduce_mft_gemma3.ipynb`](01_reproduce_mft_gemma3.ipynb)** | Build one `r=32` candidate adapter from a frozen role, save recovery checkpoints, runtime/config manifests, and FT face-sanity evidence. | OOD EM reproduction, a review decision, or Hub publication. |
| **[`02_review_candidate_adapter.ipynb`](02_review_candidate_adapter.ipynb)** | Make the matched base face-sanity bundle, blind/review candidate evidence, optionally upload a reviewed **candidate adapter**, and optionally run a disabled plumbing extraction. | OOD EM, primary RQ1, or an intervention result. |
| **[`03_ood_em_baseline.ipynb`](03_ood_em_baseline.ipynb)** | Seal the 150-text/250-VQA reconstruction; generate matched base/FT bundles; run the blinded bilateral judge; calibrate with two reviewers; seal the three-seed gate. | An exact-paper or automatically decided result. |
| **[`04_rq1_shared_residual_geometry.ipynb`](04_rq1_shared_residual_geometry.ipynb)** | Run the OOD-gated, ≥50-pair shared-residual extension per seed and strict three-seed aggregation. | The paper's final-token/SVD geometry, causal origin, or intervention efficacy. |
| [`00_colab_preflight.ipynb`](00_colab_preflight.ipynb) | Optional GPU/Drive/clone/package diagnostic. | A prerequisite or a scientific gate. |
| [`manual/verify_mft_sanity.ipynb`](manual/verify_mft_sanity.ipynb) | Manual re-check of a completed candidate adapter. | A matched base comparison or OOD EM evidence. |
| [`reference/`](reference/) | Stripped source lineage: FT, sanity, component, and synthetic generator. | Canonical data, output evidence, or a runnable/reviewed protocol. |

## Candidate-adapter loop (current)

For each seed 42, 43, and 44:

1. Run `01` through FT and face-sanity generation.
2. Run `02` through the matched base bundle and blinded candidate review.
3. Record `candidate_face_sanity_gate: pass|fail|undecided`; never call it an
   OOD EM reproduction.
4. If `pass`, optionally use the protected upload cell to persist a clearly
   labelled private candidate adapter. Keep recovery checkpoints private.

A seed-42 plumbing extraction is optional and disabled by default. It checks
the shared-residual hooks and artifact plumbing only; its 10-prompt built-in
bank cannot be inflated with repetitions or used for a primary claim.

## OOD baseline, then primary RQ1

Run `03` for seeds 42, 43, and 44 using the same sealed inputs and fixed
`evaluation_seed: 1729`:

```text
150 broad text prompts + 250 LLaVA/MSCOCO VQA pairs
matched M_base / M_ft decoder and generation seeds
balanced anonymous A/B judge order
base score + FT score + paired delta and clustered bootstrap
15 text + 25 multimodal calibration items × two reviewers
hashed per-seed review packages → hashed three-seed gate
```

The audited upstream code does not include the exact full input selections, so
this is a reconstruction, never an exact paper reproduction. The local
bilateral calibrated judge is also a project extension, not a numerically
identical copy of the upstream judge.

After all three OOD packages pass and the three-seed gate plus primary/control
prompt manifests are sealed, run `04` once per seed. The extractor validates
the exact selected adapter against its seed package, requires at least 50
unique matched prompt/image pairs, and measures paired shifts at text and
image-soft-token positions in the same Gemma language residual stream.

## Commands mirrored by the notebooks

```bash
python scripts/ft_faces.py --config <Drive-backed seed config>
python scripts/sanity_check_em.py --config <Drive-backed candidate-sanity config>
python scripts/push_adapter.py \
  --adapter-dir <FT_R32_adapter_dir> \
  --repo-id <private-hub-repo> \
  --review-summary <candidate-review-summary.json> \
  --evidence-tier candidate
python scripts/validate_ood_manifest.py <ood.jsonl> \
  --selection-rule "<frozen rule>" --reviewer "<id>" \
  --review-record "<record>" --image-root <images>
python scripts/evaluate_ood_em.py --config <materialized-ood-seed.yaml>
python scripts/judge_ood_em.py \
  --base-bundle <base.json> --ft-bundle <ft.json> \
  --pair-package <pair.json> --manifest <ood.jsonl> \
  --image-root <images> --out <judge.jsonl> --cache <cache.jsonl> \
  --judge-revision <immutable-revision> --endpoint-id <deployment-id>
python scripts/make_ood_calibration_sheet.py <required arguments>
python scripts/finalize_ood_review.py <required arguments>
python scripts/seal_ood_three_seed_gate.py <three seed reviews and decision>
python scripts/extract_rq1.py --config <materialized-primary-rq1.yaml>
python scripts/aggregate_rq1.py <seed42.json> <seed43.json> <seed44.json> \
  --out <three-seed-summary.json>
```

Exact provenance, review, and OOD requirements are in
[REPRODUCIBILITY.md](../REPRODUCIBILITY.md).
