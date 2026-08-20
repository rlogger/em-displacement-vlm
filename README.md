# Cross-Modal Emergent Misalignment: Reproduction and Displacement Protocol

Research code and a gated experimental protocol for studying whether a
fine-tuned vision-language model exhibits emergent misalignment (EM) outside
its fine-tuning domain, and—only after that is established—whether a
text-pathway intervention removes or relocates the signal.

[![Reproduce `M_ft` in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rlogger/em-displacement-vlm/blob/main/notebooks/01_reproduce_mft_gemma3.ipynb)
[![Review candidate](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rlogger/em-displacement-vlm/blob/main/notebooks/02_review_candidate_adapter.ipynb)
[![OOD baseline](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rlogger/em-displacement-vlm/blob/main/notebooks/03_ood_em_baseline.ipynb)
[![RQ1 geometry](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rlogger/em-displacement-vlm/blob/main/notebooks/04_rq1_shared_residual_geometry.ipynb)
[![Qwen2.5-VL candidate](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rlogger/em-displacement-vlm/blob/main/notebooks/01q_reproduce_mft_qwen2_5_vl_3b.ipynb)
[![Safe Drive cleanup](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rlogger/em-displacement-vlm/blob/main/notebooks/00_safe_cleanup_and_reset.ipynb)
[![Verified results](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rlogger/em-displacement-vlm/blob/main/notebooks/05_verified_results.ipynb)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Research-content warning.** The repository contains tooling for controlled
> research on harmful and stereotyped VLM outputs. It is not a deployment
> recipe or a source of demographic labels.

## Status and claim boundary

This checkout contains implementation and protocol work. It does **not**
contain an A100-produced, reviewed scientific reproduction, RQ1 result, or
BLOCK-EM result. Read [EXPERIMENT_STATUS.md](docs/EXPERIMENT_STATUS.md) before
interpreting any notebook, chart, adapter, or generated response as evidence.

The standard is deliberately sequential:

```text
provenance ledger
  -> candidate-adapter face-sanity packages for seeds 42, 43, 44
  -> reviewed OOD, paper-comparable behavioral baseline for seeds 42, 43, 44
  -> sealed extraction/evaluation prompts
  -> primary RQ1 shared-residual geometry
  -> intervention and matched controls
  -> re-discovery / displacement verification
  -> data-distribution robustness
```

A seed-42 **plumbing** extraction may be used to test the pipeline after that
seed's candidate-adapter face-sanity review. It is labelled a pilot and cannot
support an RQ1 claim. Primary RQ1 waits for all three reviewed **OOD** baseline
packages and sealed probe manifests.

## Candidate-training model lanes

- **Gemma 3-4B** remains the canonical end-to-end protocol and the only model
  family referenced by the current OOD/RQ1 notebooks.
- **Qwen2.5-VL 3B** has a separately pinned BF16 LoRA candidate-training lane
  through the shared provenance-bound runner and a CUDA 12.8 hash lock. The
  config/unit contracts and dependency resolution pass locally; real
  model/trainer construction and training remain `A100_UNRUN`. Follow
  [QWEN2_5_VL_BASELINE.md](docs/QWEN2_5_VL_BASELINE.md). Qwen adapters, reviews,
  layers, and activation artifacts must remain separate from Gemma evidence;
  Qwen OOD/RQ1 support is not yet declared complete.

## Research questions

| ID | Question | Claim boundary |
|---|---|---|
| **Candidate adapter** | Does a face fine-tune produce a coherent `M_ft` worth evaluating? | Requires paired core-image, text-only, and held-out face-sanity review; not an EM reproduction. |
| **OOD baseline** | Does `M_ft` show EM on a sealed paper-comparable reconstruction? | Requires paired `M_base`/`M_ft` broad-text and LLaVA/MSCOCO VQA evaluations plus review/judge evidence across seeds. |
| **RQ1** | Are the paired shifts at text tokens and image-soft-token positions shared or modality-specific? | Compares vectors only in the same Gemma language residual stream. |
| **RQ2** | Does a text-token intervention reduce image-conditioned behavior beyond matched controls? | Requires an implemented Gemma intervention, not the smoke model. |
| **RQ3** | If behavior remains, is it removed or relocated? | Requires frozen re-probes and re-discovery on the intervened model. |

Model states are kept distinct: `M_base` → `M_ft` (baseline subject) →
`M_abl` (inference ablation) → `M_blocked` (training-time intervention).

## Canonical workflow

1. In `01_reproduce_mft_gemma3.ipynb`, freeze the HF-backed split once with
   `data_selection_seed=42` and train one optimizer seed at `r=32`; it writes
   recovery checkpoints, a final adapter, and
   face-sanity evidence to Drive. This is a candidate-adapter check, not an EM
   reproduction, and it does **not** certify or publish the adapter.
2. In `02_review_candidate_adapter.ipynb`, generate the exact matched base
   bundle, blind/review the paired responses, record an explicit decision, and
   optionally publish a clearly labelled private candidate adapter.
3. Repeat the candidate-adapter loop for training seeds 42, 43, and 44 while
   reusing the same immutable data split. In `03_ood_em_baseline.ipynb`, build
   and review-seal a **paper-comparable** reconstruction of
   150 broad text prompts and 250 distinct-image LLaVA/MSCOCO VQA pairs.
   Generate matched
   base/FT bundles with fixed evaluation randomness, run the condition-blinded
   bilateral judge, calibrate it with two human reviewers, and seal the three
   seed packages. Exact paper selections are unavailable, so this is not an
   exact reproduction and the project judge is not numerically identical to
   the upstream metric.
4. Freeze reviewed, immutable primary/control extraction manifests. In
   `04_rq1_shared_residual_geometry.ipynb`, run the **primary** RQ1 extension
   for all three seeds and the strict aggregate.
5. Only after the three-seed RQ1 decision is reviewed, implement and evaluate
   the intervention, re-discovery, and distribution-robustness stages. Those
   stages are currently design-only; the TinyTwoTower path is only a smoke
   test and produces no production `M_abl` or `M_blocked`.

See [notebooks/README.md](notebooks/README.md) for exact notebook order and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the evidence contract.
The presentation-safe, artifact-bound result ledger is in
[RESULTS.md](docs/RESULTS.md); slide-only observations are excluded from it.

## How the team artifacts fit together

The imported FT and sanity notebooks are source lineage, not canonical
evaluation. The team synthetic-text generator is preserved as a sanitized
reference, but its own `prompt.txt` and generated prompt artifact are absent
here; it remains an unsealed candidate asset. The official upstream repository
has a distinct synthetic-generator prompt, documented in the upstream audit.

- [Team integration map and ownership](docs/TEAM_INTEGRATION_AND_ROADMAP.md)
- [Upstream protocol audit](docs/UPSTREAM_AUDIT.md)
- [Synthetic text-probe contract](docs/SYNTHETIC_TEXT_PROBES.md)
- [Behavioral-review rubric](docs/BEHAVIORAL_REVIEW.md)
- [Technical roadmap](docs/ROADMAP.md)
- [Raj's working workbook](docs/raj_tonight_workbook.html)

## Method scope

```text
Frozen harmful-faces fine-tune role (1,500 records)  -> candidate M_ft
Face-sanity bundle                                    -> candidate-adapter gate
Construction-bound OOD broad-text + 250-distinct-image VQA reconstruction -> EM baseline gate
Sealed extraction prompts                               -> paired shift vectors

c_text        = mean(h_Mft - h_Mbase) at text-token positions
c_image_token = mean(h_Mft - h_Mbase) at image-soft-token positions
```

RQ1 measures both vectors in the **same Gemma language residual stream** at
layers 20 and 32. A raw vision-tower vector is not directly comparable to a
language residual vector, and shared-residual alignment is not evidence that a
vision tower caused the behavior.

Each primary run records the model and dataset revisions, ordered split hash,
adapter fingerprint, prompt-manifest hashes, decoding settings, fixed
evaluation seed, environment versions, paired judge/calibration evidence,
three-seed review hashes, and bootstrap unit. Repeating prompts never turns a
small template bank into more independent observations.

## Repository layout

```text
em-displacement-vlm/
├── src/em_displacement_vlm/
│   ├── data/                  # immutable role manifests and contamination checks
│   ├── ft/                    # Gemma/Qwen Unsloth candidate-adapter training
│   ├── evals/                 # face sanity, OOD generation/judge/review contracts
│   ├── rq1.py                 # production shared-language-residual RQ1 path
│   ├── extraction/, interventions/
│   │                          # generic TinyTwoTower smoke helpers, not Gemma science
│   └── models/                # adapter persistence; load_model_bundle is Tiny-only
├── protocols/                 # audited sources and machine-readable stage contract
├── configs/                   # runnable base configs plus explicit design-only configs
├── scripts/                   # builders, sealers, runners, review, and persistence
├── notebooks/                 # canonical Colab notebooks and noncanonical references
├── prompts/                   # judge rubric only; real prompt banks are external/sealed
├── docs/                      # status, protocol, ownership, and roadmap
└── tests/                     # local engineering checks
```

The exact module-to-stage mapping and current implementation status are in
[ARCHITECTURE_AND_WORKFLOW.md](docs/ARCHITECTURE_AND_WORKFLOW.md) and are
validated from `protocols/workflow.yaml` in CI.

## Local engineering checks

Python 3.11+ and [uv](https://github.com/astral-sh/uv) are sufficient for
data-preparation and smoke checks; they do not reproduce the Gemma experiment.
The setup script places the environment outside the repository and links it as
`.venv`, avoiding iCloud `dataless` package placeholders on macOS.

```bash
git clone https://github.com/rlogger/em-displacement-vlm.git
cd em-displacement-vlm
./scripts/setup_local.sh
source .venv/bin/activate
uv sync --extra torch --extra vlm --extra dev

python scripts/prepare_datasets.py
python scripts/check_disjointness.py
pytest -q
python scripts/smoke_test.py --config configs/smoke.yaml
```

Those broad extras are the local engineering/processor environment, not the
Unsloth training environment. Qwen A100 work must use
[`requirements/qwen-a100.lock`](requirements/qwen-a100.lock) and the exact
construction gate in the Qwen runbook.

Large adapters, activations, and response bundles belong on Drive or the Hub,
not in git. Do not commit tokens, raw generated responses, or human-review
mappings.

## Sources and acknowledgments

- Faces EM protocol and data preparation:
  [idhantgulati/vlm-alignment](https://github.com/idhantgulati/vlm-alignment)
- Team FT, sanity, and synthetic-generator source artifacts: documented under
  [`notebooks/reference/`](notebooks/reference/)
- EM-organism patterns:
  [clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM)
- Completion-only training patterns:
  [google-gemini/gemma-cookbook](https://github.com/google-gemini/gemma-cookbook)

## License

MIT. Harmful-content datasets and generated responses are research-only.
