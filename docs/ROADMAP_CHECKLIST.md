# Qwen-first execution checklist

Status vocabulary:

- `CODE_VERIFIED`: local tests validate the implementation contract.
- `A100_UNRUN`: the production GPU path has not produced a bound artifact.
- `TEAM_REPORTED_UNVERIFIED`: a number was handed off without its replayable
  evidence package in this repository.
- `DESIGN_ONLY`: no production runner exists.

| Gate | Requirement | Status | Evidence path |
|---|---|---|---|
| G0 | Clean exact source, dedicated Qwen Drive root, pinned A100 runtime | `CODE_VERIFIED`; live session required | `00_colab_preflight.ipynb`, `requirements/qwen-a100.lock` |
| G1 | Immutable 1,500-row faces role with data seed 42 | `CODE_VERIFIED`; Drive artifact required | `scripts/prepare_datasets.py` |
| G2 | BF16 `r=32` Qwen2.5-VL 3B candidate | `CODE_VERIFIED`; `A100_UNRUN` | `01q_reproduce_mft_qwen2_5_vl_3b.ipynb` |
| G3 | Matched provenance-bound candidate review | code exists; Qwen review artifact unverified | `docs/QWEN2_5_VL_BASELINE.md` |
| G4 | Pinned/gated VLGuard JSON+ZIP acquisition | `CODE_VERIFIED`; live access required | `scripts/prepare_vlguard.py` |
| G4 | Disjoint 100 safe + 100 unsafe direction roles and 100 held-out unsafe validation images | `CODE_VERIFIED`; `A100_UNRUN` | `vlguard_vision_contrast_v1.json` on Drive |
| G4 | Dynamic Qwen image-token capture at language layer 13 | `CODE_VERIFIED`; `A100_UNRUN` | `vision_validation.py` |
| G4 | Unit direction + equal-norm seeded random control | `CODE_VERIFIED`; `A100_UNRUN` | `directions.safetensors` on Drive |
| G4 | Baseline plus repair/random at alpha 80/150/250 | `CODE_VERIFIED`; `A100_UNRUN` | `02q_vlguard_vision_validation.ipynb` |
| G4 | Primary alpha-150 refusal-ASR summary | `RESULT_UNVERIFIED` | `summary.json` on Drive |
| G5 | Qwen BLOCK-EM training/intervention runner | `DESIGN_ONLY` | no production entrypoint |
| G5 | Random/wrong-layer controls, re-discovery, capability, displacement decision | `DESIGN_ONLY` | future protocol |

The reported text screen (70 baseline, 58 primary repair, 77 random repair) is
`TEAM_REPORTED_UNVERIFIED` until its direction and generation package are
available and replayed. It is useful context for the alpha-150 registration,
not a repository-verified result.

The old Gemma OOD/RQ1 checklist is retired from the active workflow. Historical
code may be consulted for implementation lineage but cannot satisfy a Qwen gate.
