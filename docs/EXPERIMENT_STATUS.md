# Experiment status

This file is an intentionally conservative status record. A green local test
or a runnable notebook is not evidence of a completed scientific experiment.

```yaml
status_schema: 1
overall_claim_status: RESULT_UNVERIFIED
as_of: "2026-08-13"

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
    Candidate adapters and OOD *candidate pools* exist as private Hub / Drive
    artifacts (see Durable external artifacts). That is not an OOD EM result.

RESULT_UNVERIFIED:
  meaning: "No scientific conclusion may be reported from this repository snapshot."
  prohibited_claims:
    - "EM reproduced on Gemma 3-4B"
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
| Candidate LoRA `r=32` seeds 42/43/44 | Drive `checkpoints/FT_R32_gemma3_faces_seed{SEED}/`; private Hub `rlogger/FT_R32_gemma3_faces_seed{SEED}` | Built; Hub includes `review_seed{SEED}_summary.json` when pushed |
| Base model for comparison | Public `unsloth/gemma-3-4b-it` @ `bf46152c47f5dd20b896357cb51abc4c03b8ee8c` | Not re-hosted by this project |
| VQA + text OOD **candidate pools** | Hub dataset `rlogger/ood-candidates-paper-comparable-v1` (`candidates/`, `images/`); Drive `data/ood/` when present | **Dataset build done** (400 text + 400 VQA; selection seed `20260730` → 150/250) |
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
