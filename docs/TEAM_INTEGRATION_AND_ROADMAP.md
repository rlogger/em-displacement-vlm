# Team integration map and technical roadmap

This document separates source lineage, preliminary observations, controlled
evidence, and conclusions. It is the shared map for the Gemma 3-4B baseline,
RQ1, intervention, verification, and robustness work. Current scientific
status is [RESULT_UNVERIFIED](EXPERIMENT_STATUS.md).

## Claim taxonomy

| Level | Meaning | Minimum evidence |
|---|---|---|
| Source lineage | An artifact exists and was inspected. | Fingerprint and a clearly noncanonical reference copy. |
| Preliminary observation | A response or model behavior was seen. | Saved source bundle; no general claim. |
| Candidate adapter | A matched face-sanity comparison supports retaining an `M_ft`. | Frozen split, matching revisions/decoder, paired face-sanity bundles, blinded review. |
| OOD behavioral baseline | A paper-comparable reconstruction shows EM beyond the face fine-tune domain. | 150 sealed broad text prompts + 250 sealed LLaVA/MSCOCO VQA pairs; bilateral blinded scoring, two-reviewer calibration, and hashed packages across all seeds. |
| Geometry evidence | Paired internal shifts meet a pre-specified RQ1 extension rule. | Three reviewed OOD baseline packages and sealed extraction manifests. |
| Causal intervention result | A matched intervention/control experiment changes behavior. | Production Gemma intervention plus primary/random/wrong-layer controls. |

Never promote a result because a notebook ran, an adapter exists, or a single
completion is striking.

## Artifact map

| Asset | Owner / role | What is established | Correct use now | What it does **not** establish |
|---|---|---|---|---|
| `gemma3_4B_lora_faces_ft` | Sai / FT source lineage | Preserved stripped reference; source records an `r=256`, 1,600-row run. | Compare protocol details and fill compatibility ledger. | Canonical `r=32` baseline, frozen split, or rank threshold. |
| `sanity_check_ft_EM_models` | Sai / sanity source lineage | Preserved stripped reference; source loads an `r=8` adapter and samples training rows. | Design reference for probe presentation. | Matched base comparison, held-out evidence, or review decision. |
| `synthetic_text_gen_pipeline` | Sai / candidate text asset | Sanitized team-variant source is preserved with environment lookup only. It targets pooled generations and a 150-prompt output. | Candidate generator pending its own prompt, output files, hashes, and review; distinct from the official upstream generator. | A generated dataset in this repo, text-pathway evidence, or an RQ1 sample. |
| `01_reproduce_mft_gemma3.ipynb` | Raj / candidate-adapter build | Drive-backed split, recovery, FT, and FT face-sanity workflow. | Produce one seed’s candidate `M_ft`; no EM certification or push. | OOD EM reproduction. |
| `02_review_candidate_adapter.ipynb` | Raj + team / candidate validation | Matched face review/publishing path and optional guarded plumbing. | Complete each seed’s candidate review. | OOD reproduction, primary RQ1, vision-tower origin, or causal mechanism. |
| `03_ood_em_baseline.ipynb` | Raj + team / behavioral baseline | Executable manifest → generation → blinded judge → calibration → three-seed gate workflow. | Produce per-seed OOD packages and one SHA-bound gate. | Exact upstream inputs, an automatic pass, or RQ1 geometry. |
| `04_rq1_shared_residual_geometry.ipynb` | Arshia + Raj / primary RQ1 | OOD-gated ≥50-pair extraction and strict three-seed aggregation. | Run the registered shared-residual extension after G2/G3. | Paper-identical SVD geometry or a causal mechanism. |
| BLOCK-EM / re-discovery | Arshia, Sai, Satyak / future work | Tiny-model components describe intended controls. | Implement only after primary RQ1 review. | A production Gemma intervention or displacement result. |
| Distribution-A/B audit | Raj / robustness | A defined future evidence lane. | Audit source metadata/missingness and pre-specify strata before later eval. | Attribute labels inferred from faces or post-hoc subgroup claims. |

The FT and sanity source notebooks are intentionally not duplicated beyond the
stripped references. The synthetic generator is preserved at
`notebooks/reference/synthetic_text_gen_pipeline.ORIG.ipynb`; it contains no
saved candidate outputs or its team-variant `prompt.txt` specification. The
official upstream repository does include its own `syn-data-gen/prompt.txt`,
but that does not establish equivalence with the team variant.

### Source fingerprints

| Artifact | SHA-256 |
|---|---|
| Sai FT source | `ab8143b0f9e588ffa6c5bb0e073d92a3377b166884180e815ee7f648102bfe48` |
| Sai sanity source | `e4547da70abbf4ef262e11f4477955f52a0c4d8ae6a6b5c02f469870c3be76bf` |
| Sai synthetic generator | `7975d01dee2fd498dd7c2f52e6f24f02dad39192c81a98c971babca14eec4175` |

## Controlled dependency map

```text
Sai source notebooks + team observations
  -> compatibility ledger (do not mix rank / rows / seed / decoder / revision)
  -> Raj + team: candidate-adapter face-sanity packages, seeds 42 / 43 / 44
  -> all three candidate packages (optional seed-42 plumbing only)
  -> Raj + team: sealed OOD paper-comparable baseline, seeds 42 / 43 / 44
       150 broad-text prompts + 250 LLaVA/MSCOCO VQA pairs per matched protocol
  -> all three calibrated OOD packages + hashed three-seed gate
  -> Sai + Raj: sealed primary and control probe manifests
  -> Arshia: primary shared-language-residual RQ1 across all seeds
  -> Arshia: production intervention + controls
  -> Sai + Satyak: re-probe and re-discovery
  -> Raj: distribution-A audit and pre-specified distribution-B robustness
```

After seed 42’s own candidate-adapter review, an **optional, disabled-by-default plumbing
extraction** can validate storage and hooks. It must be marked
`analysis_tier: plumbing_pilot`; it cannot supply an RQ1 inference or change
the order above.

## Why RQ1 uses the shared residual stream

The primary comparison is:

```text
c_text        = mean(h_Mft - h_Mbase) at text-token positions
c_image_token = mean(h_Mft - h_Mbase) at image-soft-token positions
```

Both are in the same Gemma language residual stream at layers 20 and 32. Their
cosine and canonical-angle comparisons therefore have a common vector space.
They answer whether a shift aligns after image information reaches the language
model; they do **not** identify a raw vision tower as the causal origin.

Tower-local analyses may be useful as separate diagnostics, but no raw
vision-vector-to-language-residual cosine belongs in the canonical RQ1 claim.

## Synthetic prompts: value and limits

Once frozen, the synthetic asset can serve as a separate generic text-capability
or text-sensitivity condition. It cannot replace the EM-relevant text-only
probes, fit directions, select examples after outputs are viewed, or serve as a
new visual distribution. The full integration contract is
[SYNTHETIC_TEXT_PROBES.md](SYNTHETIC_TEXT_PROBES.md).

## Gates and ownership

| Gate | Owner(s) | Pass condition | Stop condition | Durable artifact |
|---|---|---|---|---|
| G0 — compatibility | Sai + Raj | One ledger row per source/actual run, including model/data/decoder/review provenance. | Mismatched setups collapsed into one claim. | [ledger](templates/rank_sweep_ledger.csv) |
| G1 — candidate-adapter gate | Raj + team | Frozen split, `r=32` FT, matched face-sanity bundle, blinded candidate decision. | Missing manifest, comparator, or review. | Candidate package + adapter provenance. |
| G2 — OOD baseline | Raj + team | Sealed reconstruction; fixed evaluation randomness; blinded bilateral base/FT judge; paired/clustered estimates; two-reviewer calibration. | Same-domain face results or one-sided judge scores presented as EM reproduction. | Per-seed OOD package. |
| G3 — OOD replication | Raj + team | G2 is complete for 42, 43, and 44, including negative/mixed outcomes, and every package is SHA-bound. | Omit a seed or merely declare coverage without hashes. | `ood_three_seed_gate.json`. |
| G4 — sealed extraction | Sai + Raj | Unique manifest IDs/prompts, hashes, source/review metadata, and scope fixed before outputs. | Post-output selection or pseudo-replication. | Primary/control manifests. |
| G5 — primary RQ1 | Arshia | G3/G4 complete; shared-residual geometry, nulls, and bootstrap at the prompt unit. | One-seed, face-only, or raw-tower comparison presented as conclusion. | Per-seed RQ1 bundles. |
| G6 — RQ1 decision | Arshia + team | Pre-specified three-seed aggregate is reviewed. | Selective layer/seed reporting. | Aggregate and decision record. |
| G7 — intervention | Arshia | Production Gemma runner and primary/random/wrong-layer controls. | Tiny smoke treated as execution. | `M_abl`/`M_blocked` packages. |
| G8 — verification | Sai + Satyak | Frozen re-probes, capability controls, re-discovery. | Reduced one metric called removal. | Re-discovery report. |
| G9 — robustness | Raj | Metadata/missingness audit, pre-specified image-level strata, independent Distribution B if available. | Infer identities from images or select strata post hoc. | Audit and uncertainty report. |

## Immediate deliverables

### Raj — baseline and robustness foundation

1. Complete the seed-42 candidate-adapter face-sanity package without treating
   it as an EM pass.
2. Repeat the candidate-adapter package for seeds 43 and 44; record failures
   and inconclusive outcomes too.
3. Build the sealed paper-comparable OOD reconstruction before RQ1: document
   source/selection rules for 150 broad text prompts and 250 LLaVA/MSCOCO VQA
   pairs, then obtain matched base/FT outputs, bilateral blinded scores,
   two-reviewer calibration, and explicit decisions across all seeds.
4. Build the Distribution-A audit: available source metadata, missingness, role
   counts, and exact/perceptual duplicate-risk note. Do not annotate or infer
   protected attributes from faces.
5. Hand off the three OOD provenance-linked packages and a sealed extraction
   manifest request, not an RQ1 conclusion.

### Arshia — RQ1 and intervention

1. Maintain the shared-residual extraction path and its guardrails. It is an
   extension to the paper's final-token/SVD geometry, not a claim of identical
   paper methodology.
2. Use only adapters with reviewed OOD baseline provenance plus sealed
   primary/control manifests for primary RQ1; label an earlier seed-42 run as
   plumbing.
3. Keep tower-local diagnostics distinct from the primary comparison.
4. Implement intervention and matched controls only after the primary RQ1
   decision.

### Sai and Satyak — source reconciliation and verification

1. Fill lineage/protocol details for existing runs; reconcile the reported
   rank observation in a uniform ledger.
2. Provide the synthetic source prompt, generated artifacts, metadata, and
   manual-review/removal log before it is used as an evaluation asset.
3. After a real intervention, carry out frozen re-probing, capability controls,
   and re-discovery.

## Paper-comparability lane versus extensions

Follow the original input/decoder setup as closely as the released artifacts
permit. Exact paper prompt/pair selections are unavailable, so call a sealed
reconstruction **paper-comparable**, not exact reproduction. The repository's
bilateral blinded/calibrated judge is an explicit extension and its numbers are
not the upstream judge's numbers. The paper's
default rank is `r=128`; this project's `r=32` is a controlled project anchor,
not a claimed threshold. Keep blinded review, synthetic-text sensitivity,
shared-residual geometry, tower-local diagnostics, BLOCK-EM, and distribution
robustness visibly labelled as extensions unless independently matched to the
source protocol. A rank threshold is reportable only when ledger rows share
the required protocol fields and uncertainty estimates.
