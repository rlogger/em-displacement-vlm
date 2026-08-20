# Colab notebooks

The active project is Qwen2.5-VL only. Use an A100 for both canonical experiment
notebooks and persist artifacts under
`/content/drive/MyDrive/em-displacement-vlm-qwen2-5-vl-3b`.

## Canonical order

| Order | Notebook | Produces | Does not establish |
|---|---|---|---|
| 0 | [`00_colab_preflight.ipynb`](00_colab_preflight.ipynb) | Read-only source/runtime/Drive diagnostics | A scientific result |
| 1 | [`01q_reproduce_mft_qwen2_5_vl_3b.ipynb`](01q_reproduce_mft_qwen2_5_vl_3b.ipynb) | One BF16 `r=32` Qwen candidate adapter with recovery and provenance artifacts | Emergent misalignment, vision causality, or BLOCK-EM |
| 2 | [`02q_vlguard_vision_validation.ipynb`](02q_vlguard_vision_validation.ipynb) | Pinned VLGuard roles, layer-13 vision direction, baseline/repair/random generation bundle, refusal-ASR summary | Human safety, vision-specific mechanism, BLOCK-EM, or displacement |

The second experiment notebook requires a provenance-complete adapter from the
first. It runs 100 safe and 100 unsafe direction captures, then 700 generations
over 100 disjoint held-out unsafe images. Re-running resumes the immutable
package rather than starting a new experiment.

## Utilities

| Notebook | Scope |
|---|---|
| [`00_safe_cleanup_and_reset.ipynb`](00_safe_cleanup_and_reset.ipynb) | Dry-run-first, reversible Drive archive utility. It never deletes. Use only with an explicit Qwen family/root/seed. |
| [`05_verified_results.ipynb`](05_verified_results.ipynb) | Legacy read-only artifact viewer. It does not yet summarize the new VLGuard package and is not canonical. |

## Historical notebooks

The following are retained as source lineage only and are not in the active
workflow:

- `01_reproduce_mft_gemma3.ipynb`
- `02_review_candidate_adapter.ipynb`
- `04_rq1_shared_residual_geometry.ipynb`
- `manual/verify_mft_sanity.ipynb`
- `reference/`

The old Gemma OOD baseline and OOD-pool-builder notebooks were removed. Do not
restore them into the canonical sequence or combine their Drive artifacts with
Qwen adapters, directions, or results.

## Exact active commands

Notebook `01q` runs the provenance-bound Qwen training entrypoint:

```bash
python scripts/ft_faces.py \
  --config <QWEN_DRIVE>/runs/reproduce_mft_qwen2_5_vl_3b_r32_seed42.yaml
```

Notebook `02q` seals VLGuard and executes the causal screen:

```bash
python scripts/prepare_vlguard.py \
  --root <QWEN_DRIVE>/data/vlguard \
  --direction-per-class 100 \
  --validation-unsafe 100

python scripts/validate_vlguard_vision.py \
  --config <QWEN_DRIVE>/runs/qwen_vlguard_vision_seed42.yaml \
  --validate-config-only

python -u scripts/validate_vlguard_vision.py \
  --config <QWEN_DRIVE>/runs/qwen_vlguard_vision_seed42.yaml
```

Training seed 42 here identifies the Qwen adapter. It is not the retired Gemma
seed-42 OOD experiment.

See [the Qwen baseline runbook](../docs/QWEN2_5_VL_BASELINE.md),
[the VLGuard protocol](../docs/VLGUARD_VISION_VALIDATION.md), and
[the evidence ledger](../docs/EXPERIMENT_STATUS.md).
