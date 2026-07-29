# Experiment status

This file is an intentionally conservative status record. A green local test
or a runnable notebook is not evidence of a completed scientific experiment.

```yaml
status_schema: 1
overall_claim_status: RESULT_UNVERIFIED

CODE_VERIFIED:
  meaning: "Local engineering checks can validate code paths and artifact contracts."
  evidence_required:
    - "ruff, pytest, smoke test, lockfile, and notebook-JSON checks run successfully"
  scientific_interpretation: "None; this is implementation validation only."

A100_UNRUN:
  meaning: "No repository-tracked A100 run has produced the required matched evidence package."
  missing:
    - "frozen HF-backed split manifests and environment records for seeds 42, 43, 44"
    - "completed r=32 adapters with matching provenance"
    - "matched M_base and M_ft face-sanity bundles (candidate-adapter gate only)"
    - "matched OOD bundles: 150 broad text prompts and 250 LLaVA/MSCOCO VQA pairs"
    - "bilateral blinded judge evidence with fixed evaluation randomness"
    - "two-reviewer calibration, per-seed decisions, and the SHA-bound three-seed OOD gate"

RESULT_UNVERIFIED:
  meaning: "No scientific conclusion may be reported from this repository snapshot."
  prohibited_claims:
    - "EM reproduced on Gemma 3-4B"
    - "RQ1 shared or modality-specific geometry established"
    - "BLOCK-EM removes or displaces behavior"
  next_gate: "Create candidate adapters, then a sealed paper-comparable OOD baseline across three seeds."
```

## What changes the status

| Transition | Required durable evidence |
|---|---|
| `A100_UNRUN` → candidate-adapter evidence | For each seed: immutable split manifest, run/environment manifest, completed adapter, matched face-sanity bundles, blinded review sheet/mapping, review summary, and linked adapter provenance. This is not OOD EM evidence. |
| Candidate-adapter evidence → OOD baseline | A sealed 150-text/250-VQA reconstruction; matched base/FT bundles with fixed evaluation randomness; bilateral blinded scoring; paired/clustered uncertainty; two-reviewer calibration; and explicit per-seed decisions. |
| OOD baseline → RQ1 evidence | The SHA-bound passed three-seed OOD gate, ≥50 unique reviewed primary/control prompts, the exact selected adapter/split package, and a pre-specified primary analysis config. |
| RQ1 evidence → intervention evidence | Three-seed RQ1 package, implemented Gemma intervention, and primary/random/wrong-layer controls. |
| Intervention evidence → displacement conclusion | Matched re-probes, capability controls, and re-discovery that distinguish removal from relocation. |

Record a negative, mixed, or inconclusive result with the same care as a
positive one. See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for required
fields and [TEAM_INTEGRATION_AND_ROADMAP.md](TEAM_INTEGRATION_AND_ROADMAP.md)
for ownership.
