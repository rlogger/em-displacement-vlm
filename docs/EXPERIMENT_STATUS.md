# Experiment status

This file is an intentionally conservative status record. A green local test
or a runnable notebook is not evidence of a completed scientific experiment.

```yaml
status_schema: 1
overall_claim_status: RESULT_UNVERIFIED
as_of: "2026-08-19"

CODE_VERIFIED:
  meaning: "Local engineering checks can validate code paths and artifact contracts."
  evidence_required:
    - "ruff, pytest, smoke test, lockfile, and notebook-JSON checks run successfully"
  scientific_interpretation: "None; this is implementation validation only."

A100_UNRUN:
  meaning: >
    No repository-tracked A100 package yet completes the full OOD EM gate
    (matched generation + judge + two-reviewer calibration + three-seed seal).
  still_missing_for_OOD_gate:
    - "Drive-sealed paper_comparable_ood_v1.jsonl.meta.json before generation"
    - "matched OOD generation bundles for seeds 42, 43, and 44"
    - "bilateral blinded judge evidence with fixed evaluation randomness"
    - "two-reviewer calibration, per-seed decisions, and the SHA-bound three-seed OOD gate"
  note: >
    Three public Gemma candidate adapters have immutable, reviewed face-sanity
    summaries, and OOD *candidate pools* exist on Hub / Drive (see Durable
    external artifacts). Those are candidate/input evidence, not an OOD EM result.

QWEN2_5_VL_G2:
  meaning: >
    The exact 3B model/config are pinned, the A100-targeted dependency graph is
    resolver-validated and hash-locked, and local unit/processor contracts pass.
    The real Unsloth model + LoRA + collator + trainer construction and every
    optimizer step remain A100_UNRUN.
  scientific_interpretation: "None; no Qwen adapter or behavioral evidence exists."

RESULT_UNVERIFIED:
  meaning: "No scientific conclusion may be reported from this repository snapshot."
  prohibited_claims:
    - "EM reproduced on Gemma 3-4B"
    - "EM reproduced on Qwen2.5-VL 3B"
    - "RQ1 shared or modality-specific geometry established"
    - "BLOCK-EM removes or displaces behavior"
  next_gate: >
    Rehydrate/seal the paper-comparable OOD list on Drive, then produce matched
    base/FT OOD packages across seeds 42, 43, and 44.
```

## Durable external artifacts (team Colab / Hub · Aug 2026)

These are **not** committed in git. They support engineering handoff and do
**not** clear `RESULT_UNVERIFIED`.

| Artifact | Location | Status |
|---|---|---|
| Candidate LoRA `r=32` seeds 42/43/44 | Drive `checkpoints/FT_R32_gemma3_faces_seed{SEED}/`; public Hub `rlogger/FT_R32_gemma3_faces_seed{SEED}` at revisions recorded in `protocols/external_artifacts.yaml` | Built; immutable Hub packages include passed candidate face-sanity summaries. See `docs/RESULTS.md`; this is not OOD EM. |
| Base model for comparison | Public `unsloth/gemma-3-4b-it` @ `bf46152c47f5dd20b896357cb51abc4c03b8ee8c` | Not re-hosted by this project |
| Public VQA reconstruction pool | Hub dataset `rlogger/ood-vqa-mscoco-paper-comparable` @ `4a71750a2b7a30d76abbd7ebc4cc9dd0e03d74a0` | 400-row public pool with pinned Parquet hash; candidate input, not sealed evaluation. |
| Combined VQA + text OOD candidate pool | Private Hub dataset `rlogger/ood-candidates-paper-comparable-v1`; Drive `data/ood/` when present | Authentication required; revision and file hashes remain unimported into the public ledger. |
| Unreviewed / sealed eval list | Drive `data/ood/paper_comparable_ood_v1.jsonl` (+ `.meta.json` after notebook 03 seal) | Remake from Hub pools with the same selection seed if Drive copies are missing |
| Matched OOD generation / three-seed gate | Drive `results/ood/` | Not completed |

Colab helpers: `scripts/colab_push_seed42_adapter.py`,
`scripts/colab_push_all_seed_adapters.py`.

## What changes the status

| Transition | Required durable evidence |
|---|---|
| `A100_UNRUN` → candidate-adapter evidence | For each seed: immutable split manifest, run/environment manifest, completed adapter, matched face-sanity bundles, blinded review sheet/mapping, review summary, and linked adapter provenance. This is not OOD EM evidence. |
| Candidate-adapter evidence → OOD baseline | A sealed 150-text/250-VQA reconstruction; matched base/FT bundles with fixed evaluation randomness; bilateral blinded scoring; paired/clustered uncertainty; two-reviewer calibration; and explicit per-seed decisions. |
| OOD baseline → RQ1 evidence | The SHA-bound passed three-seed OOD gate, ≥50 unique reviewed primary/control prompts, the exact selected adapter/split package, and a pre-specified primary analysis config. |
| RQ1 evidence → intervention evidence | Three-seed RQ1 package, implemented Gemma intervention, and primary/random/wrong-layer controls. |
| Intervention evidence → displacement conclusion | Matched re-probes, capability controls, and re-discovery that distinguish removal from relocation. |
| Later · distribution stress-test | Pre-specified Dist B + Block-EM transfer / re-discovery (roadmap G8–G9); design-only until intervention exists. |

Record a negative, mixed, or inconclusive result with the same care as a
positive one. See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for required
fields and [TEAM_INTEGRATION_AND_ROADMAP.md](TEAM_INTEGRATION_AND_ROADMAP.md)
for ownership.
