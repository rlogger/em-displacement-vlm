# Qwen-first execution checklist

Status vocabulary:

- `CODE_VERIFIED`: local tests validate the implementation contract.
- `A100_UNRUN`: the production GPU path has not produced a bound artifact.
- `TEAM_REPORTED_UNVERIFIED`: a number was handed off without its replayable
  evidence package in this repository.
- `BLOCKED_MISSING_TEXT_PACKAGE`: Step 3 cannot run until the text tensor and
  all source/construction/generation bindings validate.
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
| G4 | Replayable unit vision direction, construction activations, and random control | `CODE_VERIFIED`; `A100_UNRUN` | vision `direction_package.json` on Drive |
| G4 | Baseline plus repair/random at alpha 80/150/250 | `CODE_VERIFIED`; `A100_UNRUN` | `02q_vlguard_vision_validation.ipynb` |
| G4 | Primary alpha-150 refusal-ASR summary | `RESULT_UNVERIFIED` | vision `summary.json` on Drive |
| G5 | Replayable Qwen text direction and bound source/generation package | `TEAM_REPORTED_UNVERIFIED`; package missing | text `direction_package.json` on Drive |
| G5 | Exact model/adapter/layer/site compatibility and construction replay | `BLOCKED_MISSING_TEXT_PACKAGE` | `QWEN_CROSS_PATHWAY_COMPARISON.md` |
| G5 | Signed geometry with bootstrap, permutation, and stability controls | `RESULT_UNVERIFIED` | `geometry_summary.json` on Drive |
| G5 | Common held-out baseline and 2 x 2 direction/site matrix | `BLOCKED_MISSING_TEXT_PACKAGE`; `A100_UNRUN` | `03q_qwen_cross_pathway_comparison.ipynb` |
| G5 | Simultaneous `own_path_both` plus matched individual and combined random controls | `BLOCKED_MISSING_TEXT_PACKAGE`; `A100_UNRUN` | `cross_pathway_summary.json` on Drive |
| G6 | Qwen BLOCK-EM training/intervention runner | `DESIGN_ONLY` | no production entrypoint |
| G6 | Random/wrong-layer controls, re-discovery, capability, displacement decision | `DESIGN_ONLY` | future protocol |

The reported text screen (70 baseline, 58 primary repair, 77 random repair) is
`TEAM_REPORTED_UNVERIFIED` until its direction and generation package are
available and replayed. It is useful context for the alpha-150 registration,
not a repository-verified result and not permission to run Step 3. The text
package must match the exact reviewed Qwen adapter used by the vision package.

The Step 3 primary compares directions within a token site at alpha 150. Raw
effects across text and vision sites are not a pathway-strength comparison
because the dynamic masks have different token counts. Even a complete Step 3
keyword screen does not clear G6; BLOCK-EM and displacement remain design-only.

The old Gemma OOD/RQ1 checklist is retired from the active workflow. Historical
code may be consulted for implementation lineage but cannot satisfy a Qwen gate.
