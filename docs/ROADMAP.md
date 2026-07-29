# Technical roadmap

Technical gates—not calendar estimates—control progress. The repository is at
implementation validation; see [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md).
Protocol facts versus project extensions are separated in
[UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md).

```text
G0  provenance + compatibility ledger
G1  candidate-adapter face-sanity, seeds 42 / 43 / 44
G2  sealed OOD paper-comparable baseline, seeds 42 / 43 / 44
G3  sealed RQ1 extraction inputs
G4  primary three-seed shared-residual RQ1 extension
G5  production intervention + controls
G6  re-discovery / displacement verification
G7  data-distribution robustness
```

## G0 — provenance and comparability

**Goal:** make every source observation and new run comparable only where its
fields truly match.

1. Record rank, alpha, seed, model revision, dataset revision, ordered split
   hash, decoder, metric/judge, environment, review, and uncertainty in the
   [ledger](templates/rank_sweep_ledger.csv).
2. Pin upstream protocol facts to
   [`idhantgulati/vlm-alignment` @ `84bfc695`](https://github.com/idhantgulati/vlm-alignment/tree/84bfc695386ba56c6740eb7c00a8481830ac1c34).
3. Keep the paper default `r=128` distinct from the project’s `r=32` anchor.
   A rank threshold is a hypothesis until a matched grid demonstrates it.

**Stop:** do not summarize `r=8`, `r=32`, `r=128`, or `r=256` observations as
one rank result.

## G1 — candidate-adapter face-sanity

**Goal:** create a reproducible `M_ft` candidate for each seed, not yet an EM
reproduction.

1. Freeze the 1,500-row harmful-faces induction role with the HF-backed route.
2. Fine-tune Gemma 3-4B at the project anchor `r=32` for seeds 42, 43, and 44.
3. For each seed, generate matched `M_base`/`M_ft` face-sanity evidence:
   core-image, text-only bleed-through, and held-out face batch.
4. Blind/review all responses and save an explicit
   `candidate_face_sanity_gate: pass|fail|undecided` decision with the adapter.
5. Privately persist only reviewed candidate adapters, with their review
   summary and provenance. A pass permits an optional plumbing extraction.

The face role is still in-domain for the visual fine-tune. It is useful
engineering and screening evidence, but it cannot establish OOD EM.

**Optional seed-42 plumbing:** run only after that seed’s candidate review,
with `analysis_tier: plumbing_pilot`. It validates hooks, manifests, and
storage; no RQ1 inference is permitted.

## G2 — OOD paper-comparable behavioral baseline

**Goal:** test whether the candidate adapters show the paper’s target type of
behavior beyond the fine-tune domain.

The target evaluation is 150 broad text prompts plus 250 LLaVA/MSCOCO VQA
pairs. Exact upstream input selections and generated 150/250 assets are not
released in the audited source, so create a **sealed paper-comparable
reconstruction**, never an “exact reproduction.”

For each seed:

1. Write source/selection rules before opening model outputs.
2. Seal unique prompt/pair manifests with IDs and SHA-256 values.
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

**Go to G3 only if:** all three seeds have reviewed OOD packages. Negative or
mixed results remain results; they are not grounds to omit a seed.

## G3 — sealed RQ1 extraction inputs

**Goal:** prevent post-output prompt selection and pseudo-replication.

1. Freeze primary text and image-conditioned extraction manifests independent
   of G2’s evaluation outputs.
2. Require unique normalized IDs/prompts, review metadata, hashes, and
   pre-specified bootstrap unit.
3. Bind the RQ1 configuration to the selected adapter fingerprint, ordered
   split, and the hashed three-seed OOD gate. A declaration of seed coverage
   without the three underlying review-package hashes is invalid.
4. Use at least 50 unique matched prompt/image pairs for a primary run. The
   built-in ten-prompt bank remains plumbing only.

The candidate synthetic-text bank is only a separately scoped sensitivity
asset after it meets [its contract](SYNTHETIC_TEXT_PROBES.md). It does not
replace the primary OOD reconstruction.

## G4 — primary RQ1 shared-residual geometry extension

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
- Aggregate only the pre-specified three seed packages.

This is a project extension; it is not the paper’s final-token/SVD geometry.
Never compare raw vision-tower vectors directly with language residuals or call
shared-residual alignment proof of vision-tower causality.
The orientation-reference tail fraction is not a p-value and is not part of
the cross-seed decision rule.

## G5 — intervention and displacement

**Goal:** evaluate a production Gemma intervention only after the RQ1 decision.

```text
L = L_task + lambda * ||proj_c_text(h)||^2
```

Use a text-token primary arm and matched random-equal-norm / wrong-layer
controls. TinyTwoTower smoke is engineering validation, not this experiment.

## G6 — verification and re-discovery

**Goal:** distinguish removal from relocation.

Use the same sealed probe families and capability controls. If an intervention
changes text behavior but visual behavior remains, re-discover on the intervened
model before calling the effect displacement or removal.

## G7 — data-distribution robustness

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
