# Upstream protocol audit

The authoritative machine-readable record is
[`protocols/upstream_sources.yaml`](../protocols/upstream_sources.yaml). This
page is its human-readable interpretation.

| Item | Audited source | Safe use in this repository |
|---|---|---|
| Official implementation | [`idhantgulati/vlm-alignment` @ `84bfc695`](https://github.com/idhantgulati/vlm-alignment/tree/84bfc695386ba56c6740eb7c00a8481830ac1c34) | Inspect and link protocol details. No LICENSE was found in the audited checkout, so upstream code is not copied here. |
| Official FT / judge / subspace / synthetic-prompt code | Same pinned commit | Source for a paper-comparable reconstruction, subject to a separate protocol ledger. |
| Exact OOD input assets | Not found in the audited upstream checkout beyond `em-judge/io/input-sample.json` | Do not call a reconstructed 150/250 evaluation the exact paper reproduction. |
| Paper | [arXiv:2602.16931](https://arxiv.org/abs/2602.16931), [OpenReview PDF](https://openreview.net/pdf/28d7af9dff30ba1867ef9e1ac65e7cecbe8c8bf4.pdf) | The paper's OOD evaluation target is 150 broad text prompts plus 250 LLaVA/MSCOCO VQA pairs; its default rank is `r=128`. |
| This repository | Project-specific extensions | `r=32` is a project anchor; face-sanity is a candidate-adapter check; shared-residual RQ1 is an extension to the paper's final-token/SVD geometry. |

The executable OOD lane matches the released input counts and decoder where
possible, but its exact A/B bilateral scoring, paired/clustered uncertainty,
and two-reviewer calibration are project extensions. Those numbers are not
the upstream judge's numerically identical metric.

The team synthetic generator is a distinct variant. Its sanitized,
output-free source reference is retained for provenance, but its own
`prompt.txt`, generated prompt outputs, and manual-review log are not present.
See [SYNTHETIC_TEXT_PROBES.md](SYNTHETIC_TEXT_PROBES.md).
