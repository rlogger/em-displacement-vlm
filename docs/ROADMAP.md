# Technical roadmap

Technical gates—not calendar estimates—control progress. The repository is at
implementation validation; see [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md).
Protocol facts versus project extensions are separated in
[UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md).

```text
G0  repository/runtime preflight
G1  frozen data + provenance
G2  candidate M_ft training, seeds 42 / 43 / 44
G3  candidate face-sanity review
G4  sealed OOD EM baseline, seeds 42 / 43 / 44
G5  sealed RQ1 probe banks
G6  primary three-seed shared-residual RQ1 extension
G7  production intervention + controls
G8  re-discovery / displacement verification
G9  data-distribution robustness
```

These IDs are identical to the authoritative
[`protocols/workflow.yaml`](../protocols/workflow.yaml) contract.

## G0 — repository/runtime preflight

**Goal:** establish a known code revision and compatible execution environment
without mistaking structural readiness for a scientific result.

1. Run the local/Colab preflight and record repository commit, GPU, Python,
   Torch/CUDA, dependency versions, secret presence, and writable persistence.
2. Pin upstream protocol facts to
   [`idhantgulati/vlm-alignment` @ `84bfc695`](https://github.com/idhantgulati/vlm-alignment/tree/84bfc695386ba56c6740eb7c00a8481830ac1c34).
3. Record every sourced or actual run in the
   [ledger](../templates/rank_sweep_ledger.csv) without combining incomparable
   model/data/decoder/review protocols.

**Stop:** do not summarize `r=8`, `r=32`, `r=128`, or `r=256` observations as
one rank result. The paper default is `r=128`; `r=32` is this project’s anchor,
not a demonstrated threshold.

## G1 — frozen data and provenance

**Goal:** freeze the real HF-backed roles once so later training-seed
comparisons do not also change the sampled data.

1. Materialize the 1,500-row harmful-faces induction role with
   `data_selection_seed: 42`.
2. Bind the dataset ID/revision, ordered row identities and hashes, role
   counts, and disjointness evidence in an immutable split manifest.
3. Reuse this exact split for training seeds 42, 43, and 44. Offline fixtures
   may exercise plumbing but cannot clear G1.

## G2 — candidate `M_ft` training

**Goal:** create a reproducible `M_ft` candidate for each seed, not yet an EM
reproduction.

1. Fine-tune Gemma 3-4B at the project anchor `r=32` for optimizer/training
   seeds 42, 43, and 44, reusing that exact role. This estimates training-seed
   variation conditional on one data selection instead of confounding two
   sources of randomness.
2. Preserve response-only label-mask audit, recovery checkpoints, effective
   config, environment, split hash, and immutable reproduction manifest.
3. Save a local candidate adapter without calling training completion an EM
   result.

## G3 — candidate face-sanity review

**Goal:** screen each candidate on matched in-domain evidence before spending
the OOD evaluation budget.

1. For each seed, generate matched `M_base`/`M_ft` face-sanity evidence:
   core-image, text-only bleed-through, and held-out face batch.
2. Blind/review all responses and save an explicit
   `candidate_face_sanity_gate: pass|fail|undecided` decision with the adapter.
3. Privately persist only reviewed candidate adapters, with their review
   summary and provenance. A pass permits an optional plumbing extraction.

The face role is still in-domain for the visual fine-tune. It is useful
engineering and screening evidence, but it cannot establish OOD EM.

**Optional seed-42 plumbing:** run only after that seed’s candidate review,
with `analysis_tier: plumbing_pilot`. It validates hooks, manifests, and
storage; no RQ1 inference is permitted.

## G4 — OOD paper-comparable behavioral baseline

**Goal:** test whether the candidate adapters show the paper’s target type of
behavior beyond the fine-tune domain.

The target evaluation is 150 broad text prompts plus 250 LLaVA/MSCOCO VQA
pairs. Exact upstream input selections and generated 150/250 assets are not
released in the audited source, so create a **sealed paper-comparable
reconstruction**, never an “exact reproduction.”

For each seed:

1. Supply pinned candidate source manifests and write source/selection rules
   before opening model outputs.
2. Use `build_ood_manifest.py` for deterministic source-item selection, then
   separately review and seal unique prompt/pair manifests with dataset
   revision, source item IDs, image hashes, and the registered rule of 250
   distinct images for 250 multimodal pairs.
3. Generate matched base/FT outputs with identical decoding and one fixed
   evaluation seed shared across adapter seeds 42/43/44.
4. Score all three responses for both conditions under an exactly balanced,
   blinded A/B assignment. Report base, FT, paired delta, event-rate
   difference, coherence, mean-of-three sensitivity, and paired
   prompt/image-cluster bootstrap intervals.
5. Calibrate on 15 text and 25 multimodal items with two independent blinded
   reviewers. Generation and automated judge artifacts remain `undecided`.
6. Record the per-seed `ood_em_reproduction_gate:
   pass|fail|undecided`, then SHA-bind all three seed packages into one
   cross-seed gate. The project judge is not numerically identical to the
   upstream judge.

**Go to G5 only if:** all three seeds have reviewed OOD packages. Negative or
mixed results remain results; they are not grounds to omit a seed.

## G5 — sealed RQ1 extraction inputs

**Goal:** prevent post-output prompt selection and pseudo-replication.

1. Freeze primary text and image-conditioned extraction manifests independent
   of G4’s evaluation outputs.
2. Require unique normalized IDs/prompts, explicit ordered `pair_id` matching
   across EM/control banks, review metadata, hashes, and a pre-specified
   bootstrap unit.
3. Bind the RQ1 configuration to the selected adapter fingerprint, ordered
   split, and the hashed three-seed OOD gate. A declaration of seed coverage
   without the three underlying review-package hashes is invalid.
4. Use at least 50 unique matched prompt/image pairs for a primary run. The
   built-in ten-prompt bank remains plumbing only.

The candidate synthetic-text bank is only a separately scoped sensitivity
asset after it meets [its contract](SYNTHETIC_TEXT_PROBES.md). It does not
replace the primary OOD reconstruction.

## G6 — primary RQ1 shared-residual geometry extension

**Goal:** test whether paired FT shifts align at text-token and image-soft-token
positions in a common Gemma language residual space.

```text
c_text        = mean(h_Mft - h_Mbase) at text-token positions
c_image_token = mean(h_Mft - h_Mbase) at image-soft-token positions
```

- Capture layers 20 and 32 in the same language residual stream.
- Report per-seed cosine, paired confidence interval, descriptive random
  equal-norm orientation reference, and canonical angles at the
  independent-prompt level.
- Aggregate only the pre-specified three seed packages. The registered
  conclusion uses the prompt-paired primary-minus-control cosine contrast;
  primary and control cosines alone remain descriptive.

This is a project extension; it is not the paper’s final-token/SVD geometry.
Never compare raw vision-tower vectors directly with language residuals or call
shared-residual alignment proof of vision-tower causality.
The orientation-reference tail fraction is not a p-value and is not part of
the cross-seed decision rule.

An observed text/image overlap is descriptive, not a modality-lead result.
Modality leadership is a later causal question: compare symmetric,
modality-restricted interventions under the same sealed behavioral and
capability evaluations. Do not infer “vision leads” merely because the input
begins with an image or adapters include vision modules.

## G7 — intervention and displacement

**Goal:** evaluate a production Gemma intervention only after the RQ1 decision.

```text
L = L_task + lambda * ||proj_c_text(h)||^2
```

Use a text-token primary arm and matched random-equal-norm / wrong-layer
controls. TinyTwoTower smoke is engineering validation, not this experiment.
`configs/block_em_design.yaml` records the intended sweep but has no production
runner; it cannot create an intervention result.

## G8 — verification and re-discovery

**Goal:** distinguish removal from relocation.

Use the same sealed probe families and capability controls. If an intervention
changes text behavior but visual behavior remains, re-discover on the intervened
model before calling the effect displacement or removal.

## G9 — data-distribution robustness

**Goal:** determine whether findings are stable across planned data conditions.

1. Audit Distribution A’s source-provided metadata availability, missingness,
   role counts, and exact/perceptual duplicate risk.
2. Pre-specify image-level strata without inferring protected attributes from
   faces.
3. If a second visual distribution is available, define it before evaluation
   and report it separately.

## Persistence and run records

- **GitHub:** code, configs, manifests, documentation, and checks.
- **Drive / private Hub:** adapters, activation artifacts, review packages, and
  recovery checkpoints.
- **Every actual run:** git commit, config hash, seed, environment manifest,
  model/data/prompt revisions, output hashes, and final status.

The notebook flow and command equivalents are in
[notebooks/README.md](../notebooks/README.md); detailed provenance is in
[REPRODUCIBILITY.md](../REPRODUCIBILITY.md).
