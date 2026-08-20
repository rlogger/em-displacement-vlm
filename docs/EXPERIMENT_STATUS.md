# Experiment status

Last repository audit: 2026-08-20.

## Active scientific status

```yaml
project_focus: qwen2_5_vl_3b
persistent_root: /content/drive/MyDrive/em-displacement-vlm-qwen2-5-vl-3b

qwen_candidate_training:
  code_status: CODE_VERIFIED
  runtime_status: A100_UNRUN
  result_status: RESULT_UNVERIFIED
  required_result: provenance-complete BF16 r32 adapter plus matched review

text_causal_validation:
  status: TEAM_REPORTED_UNVERIFIED
  reported_only:
    baseline_asr_percent: 70
    primary_repair_asr_percent: 58
    random_repair_asr_percent: 77
  missing_from_public_main:
    - bound direction tensor
    - immutable generation bundle
    - judge/config/runtime summary
  scientific_interpretation: none until imported and replayed

vlguard_vision_validation:
  code_status: CODE_VERIFIED
  runtime_status: A100_UNRUN
  result_status: RESULT_UNVERIFIED
  registered_site: qwen language residual layer 13 at dynamic image-token positions
  primary_alpha: 150
  sensitivity_alphas: [80, 250]
  required_controls: [baseline, equal_norm_seeded_random]

qwen_blocking_and_displacement:
  implementation_status: DESIGN_ONLY
  runtime_status: A100_UNRUN
  result_status: RESULT_UNVERIFIED
```

`CODE_VERIFIED` means local unit/contract tests pass. It never means that a
model was trained or that a causal effect was observed.

## What is no longer active

The Gemma seed-42 OOD experiment and its notebook are not part of the current
project. The OOD baseline and OOD-pool-builder Colabs were removed from the
repository. Historical Gemma adapters, modules, and external artifacts remain
lineage only; this repository change did not delete anything from Drive or the
Hugging Face Hub.

## Evidence required for a reportable vision result

The strongest allowed label before all items exist is `RESULT_UNVERIFIED`.

| Required item | Current repository state |
|---|---|
| Exact Qwen adapter identity/revision/seed and content fingerprint | Validator implemented; no run artifact checked in |
| Pinned VLGuard revision and accepted access terms | Revision registered; gated download requires the user's token |
| Hash-sealed disjoint direction and validation roles | Builder/validator implemented; Drive manifest not yet produced |
| Unit layer-13 direction and equal-norm random tensor | Capture/save/replay implemented; A100 artifact absent |
| Complete baseline/repair/random bundle | Row-resumable runner implemented; A100 artifact absent |
| Primary alpha-150 refusal-ASR comparison | Summary implementation exists; no measured summary yet |
| Human review or a stronger judge | Not part of the current keyword-screen gate |

Once the A100 run completes, it may be described narrowly as a
`MEASURED_VLGUARD_KEYWORD_SCREEN`. It must name the model, adapter, manifest,
commit, layer, alpha, random control, and keyword judge. It must not be called a
human safety result, successful BLOCK-EM, or displacement.

## Next executable step

1. Run `notebooks/01q_reproduce_mft_qwen2_5_vl_3b.ipynb` on A100 if the Qwen
   adapter does not already exist under the dedicated Drive root.
2. Complete the matched Qwen candidate review described in
   `docs/QWEN2_5_VL_BASELINE.md`.
3. Run `notebooks/02q_vlguard_vision_validation.ipynb` on A100.
4. Preserve `run_metadata.json`, `directions.safetensors`,
   `direction_metadata.json`, `generations.jsonl`, and `summary.json` together.
5. Only after reviewing that package, specify and implement the production Qwen
   BLOCK-EM gate.
