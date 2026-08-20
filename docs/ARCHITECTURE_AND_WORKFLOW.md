# Architecture and Workflow Contract

The authoritative machine-readable workflow is
[`protocols/workflow.yaml`](../protocols/workflow.yaml). It records what is
runnable, what still requires manual inputs, and what remains design-only.
Passing its validator means only that the declaration is structurally
consistent. It never means that training ran, reviewers approved evidence, an
experimental gate passed, or a scientific claim was established.

Run the structural check with:

```bash
python scripts/validate_workflow.py
```

The validator reads YAML, inspects source-controlled paths and Python symbol
declarations, and exits. It does not import the fine-tuning, evaluation, RQ1,
model, extraction, or intervention implementations. It does not load a model,
dataset, checkpoint, secret, notebook, judge endpoint, or experiment output.

## Gate architecture

| Gate | Capability | Status | Required predecessor | Meaning |
|---|---|---|---|---|
| G0 | Repository and runtime preflight | `runnable` | — | Record compatibility and structural readiness. |
| G1 | Frozen data and provenance | `runnable` | G0 | Freeze the real HF-backed fine-tune role and hashes. |
| G2 | Candidate `M_ft` training | `runnable` | G1 | Train one provenance-bound candidate adapter. |
| G3 | Candidate face-sanity review | `manual_inputs_required` | G2 | Human-review matched face-domain evidence; this is not OOD EM. |
| G4 | OOD EM reconstruction gate | `manual_inputs_required` | G3 | Evaluate all three seeds on one construction-bound 150-text/250-distinct-image reconstruction. |
| G5 | Sealed RQ1 probe banks | `manual_inputs_required` | G4 | Approve explicit-pair-ID primary/control banks before activation inspection. |
| G6 | Shared-residual RQ1 geometry | `runnable` | G5 | Run the same-space three-seed extension with a registered primary-minus-control contrast. |
| G7 | Production intervention | `design_only` | G6 | No production Gemma `M_abl` or `M_blocked` runner exists yet. |
| G8 | Re-discovery and displacement | `design_only` | G7 | No production re-discovery or relocation diagnostic exists yet. |
| G9 | Distribution robustness | `design_only` | G8 | No sealed distribution-B evaluation exists yet. |

These are dependency gates, not calendar phases. A later gate cannot be cleared
by code presence alone, and local smoke tests cannot substitute for a required
GPU run, immutable artifact, or review decision.

## Status semantics

- `runnable` means that a production entrypoint exists. It says nothing about
  whether runtime dependencies are available or a result has been produced.
- `manual_inputs_required` means repository support exists, but frozen external
  inputs, credentials, or durable human review are also required. G4 includes
  a deterministic candidate-manifest builder; G5 includes the EM/control bank
  sealer, but neither invents or approves the missing research inputs.
- `design_only` means evidence requirements are specified but production code
  does not exist. A config, tensor helper, or roadmap entry is not an
  implementation.

The contract deliberately locks G7–G9 to `design_only`. Moving one of them to a
different status requires both a production implementation and an intentional
validator update; editing the YAML alone fails CI.

## Canonical notebook ownership

Each canonical notebook belongs to exactly one gate:

| Notebook | Gate |
|---|---|
| `00_colab_preflight.ipynb` | G0 |
| `01_reproduce_mft_gemma3.ipynb` | G2 |
| `02_review_candidate_adapter.ipynb` | G3 |
| `03_ood_em_baseline.ipynb` | G4 |
| `04_rq1_shared_residual_geometry.ipynb` | G6 |

Reference and manual notebooks are not canonical workflow entrypoints. CI
rejects a missing, reordered, duplicated, or multiply owned canonical notebook.
`00_safe_cleanup_and_reset.ipynb`, `01q_reproduce_mft_qwen2_5_vl_3b.ipynb`,
and `05_verified_results.ipynb` are explicitly non-canonical utilities/model
lanes: cleanup only archives, and results inspection only verifies existing
evidence. Neither operation changes the scientific workflow status.

## Production and smoke boundaries

The production paths are intentionally narrower than the package tree:

- Gemma and Qwen2.5-VL candidate training use `scripts/ft_faces.py` and
  `src/em_displacement_vlm/ft`, with separate pinned configs and artifact
  namespaces. Qwen additionally requires the platform-specific
  `requirements/qwen-a100.lock`; only its A100-targeted dependency graph has
  been resolver-validated. The real-construction gate has not yet run on A100.
  The canonical notebooks and downstream OOD/RQ1 path remain Gemma-specific;
  the Qwen G2 runbook is `docs/QWEN2_5_VL_BASELINE.md`.
- OOD generation, blinded judging, calibration, and gate sealing use the
  dedicated OOD scripts, `src/em_displacement_vlm/evals/candidate_review.py`,
  `src/em_displacement_vlm/evals/ood_em.py`, and
  `src/em_displacement_vlm/evals/ood_review.py`. The OOD runner refuses an
  adapter without its exact passed G3 summary. Every judge row is content-bound
  to the saved base/FT responses; per-seed and three-seed reviews are replayed
  from those immutable bundles before RQ1 or its aggregate can consume them.
- Primary RQ1 uses `scripts/extract_rq1.py`,
  `scripts/aggregate_rq1.py`, and `src/em_displacement_vlm/rq1.py`.

The following generic components are declared smoke-only:

- token-position extraction in `src/em_displacement_vlm/extraction`;
- tensor intervention and `BlockEMTrainerStep` skeletons in
  `src/em_displacement_vlm/interventions`;
- `load_model_bundle` in `src/em_displacement_vlm/models`.

That model-module boundary applies to the generic loader, not to its independent
adapter persistence helpers. The workflow validator verifies these declared
symbols without importing them.

## Artifact versus path declarations

`paths` are source-controlled files and must exist when CI runs. `artifacts`
name the evidence classes a gate is expected to produce at runtime; they are not
asserted to exist by structural validation. This separation prevents a clean
repository checkout from being mistaken for completed research while still
making the required evidence machine-readable.

The contract is therefore a truthful map of current capability:

```text
implemented baseline and RQ1 prefix
    -> manual scientific execution and review
    -> design-only intervention
    -> design-only re-discovery
    -> design-only robustness
```
