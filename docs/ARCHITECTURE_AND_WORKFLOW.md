# Architecture and workflow

The machine-readable source of truth is `protocols/workflow.yaml`. CI validates
its order, declared files, design-only boundary, and canonical notebook list.
That structural pass never runs a model or establishes a scientific result.

## Active gate graph

```text
G0 repository preflight
 |
G1 frozen Qwen faces data
 |
G2 Qwen candidate training
 |
G3 matched Qwen candidate review
 |
G4 VLGuard vision direction package and own-path validation
 |
G5 Step 3: replayable text input + cross-pathway comparison
 |
G6 Qwen BLOCK-EM and displacement [DESIGN ONLY]
```

| Gate | Production surface | Durable output |
|---|---|---|
| G0 | `00_colab_preflight.ipynb`, CI, local setup | commit/runtime/root record |
| G1 | `prepare_datasets.py`, `data/` | immutable split and hashes |
| G2 | `01q_reproduce_mft_qwen2_5_vl_3b.ipynb`, `ft_faces.py`, `ft/` | adapter, checkpoints, config/runtime manifests |
| G3 | sanity, annotation, and summary scripts | matched base/FT bundles and review summary |
| G4 | `02q_vlguard_vision_validation.ipynb`, `prepare_vlguard.py`, `validate_vlguard_vision.py`, `vision_validation.py` | sealed roles, replayable vision package, generations, ASR summary |
| G5 | `03q_qwen_cross_pathway_comparison.ipynb`, `compare_qwen_pathways.py`, `cross_pathway.py` | replayed text/vision inputs, geometry, common held-out causal matrix |
| G6 | `block_em_design.yaml`, generic smoke helpers | none; production runner absent |

## Qwen Drive boundary

All active artifacts live below exactly:

```text
/content/drive/MyDrive/em-displacement-vlm-qwen2-5-vl-3b/
  data/splits/seed42/
  data/vlguard/
    images/
    vlguard_vision_contrast_v1.json
  checkpoints/
    training/FT_R32_qwen2_5_vl_3b_faces_seed<SEED>/
    FT_R32_qwen2_5_vl_3b_faces_seed<SEED>/
  runs/
  results/vlguard_vision/seed<SEED>/
  results/text_direction/seed<SEED>/
  results/cross_pathway/seed<SEED>/
  cache/huggingface/
```

Model families are not interchangeable. A Gemma adapter, review, activation,
or result never satisfies a Qwen gate. The Qwen training seed may be 42, 43, or
44; this identifier is unrelated to the retired Gemma seed-42 OOD experiment.

## Vision-validation data flow

```text
pinned VLGuard train.json + train.zip
        |
strict schema/archive validation
        |
hash-ranked image-disjoint roles
  100 safe direction images
  100 unsafe direction images
  100 unsafe validation images
        |
Qwen layer-13 residuals at dynamic image-token positions
        |
unit mean(unsafe)-mean(safe) direction + equal-norm random
        |
held-out original unsafe instructions
  baseline
  repair alpha 80 / 150 / 250
  random alpha 80 / 150 / 250
        |
row-resumable private generation bundle
        |
deterministic refusal-ASR summary
        |
replayable vision direction package
```

The layer hook applies steering only during the prefill pass whose sequence
matches the processor-derived image mask. Decode steps are not globally shifted.
No fixed image-token index or fixed image-token count is used.

## Step 3 cross-pathway data flow

```text
replayable text package       replayable VLGuard vision package
          |                                |
          +-- exact same model / adapter / layer / residual site --+
                                      |
                            replay construction tensors
                                      |
                signed cosine + bootstrap/stability controls
                                      |
                common held-out VLGuard validation manifest
                                      |
 baseline + 2 x 2 direction/site matrix + own-path-both
               + matched same-site random controls
                                      |
                    private rows + paired ASR summary
```

The four matrix cells are `text_at_text`, `text_at_vision`,
`vision_at_text`, and `vision_at_vision`. `own_path_both` simultaneously
applies `-c_text` at text positions and `-c_vis` at image positions. Random
conditions use the same direction source/site naming with a `random_` prefix;
`random_both_own` controls the combined arm.

All conditions reuse the exact same held-out image/prompt rows, decoding,
judge, and item identities. Alpha 150 is the primary per-token scale. Because
the two masks contain different token counts, effects may be compared between
directions within a site but not interpreted across sites as pathway strength.
See [QWEN_CROSS_PATHWAY_COMPARISON.md](QWEN_CROSS_PATHWAY_COMPARISON.md).

## Provenance boundaries

The runner refuses:

- a dirty Git worktree or runtime that differs from the A100 hash lock;
- an adapter with another model ID, revision, training seed, or manifest hash;
- an unregistered VLGuard revision or changed selected-image bytes;
- overlapping direction/validation roles;
- zero/non-finite directions or unequal random-control norms;
- partial or cross-run outputs.

Step 3 additionally refuses a missing text package; model/adapter/layer/site or
hook-semantic mismatches between packages; directions that do not replay from
construction activations; direction/evaluation leakage; changed common
held-out rows; empty or overlapping token masks; and an incomplete
direction/site/random condition set.

The handed-off text figures 70/58/77 are `TEAM_REPORTED_UNVERIFIED`, not a text
package. The final vision or cross-pathway refusal-ASR summary is still a
heuristic causal screen. Neither is a human-reviewed safety evaluation, proof
of a vision-specific mechanism, or a BLOCK-EM/displacement result. G6 remains
design-only even after G5 executes.

## Retired and historical surfaces

The Gemma OOD notebooks were removed from the active repository workflow.
Gemma training, review, OOD, and RQ1 modules remain for provenance/history and
tests, but are not canonical Qwen entrypoints. `04_rq1_shared_residual_geometry`
and `05_verified_results` are explicitly noncanonical legacy notebooks.

The generic `extraction/`, `interventions/`, and `models/` components remain
smoke-only. Their TinyTwoTower tests cannot clear G4 or G5.
