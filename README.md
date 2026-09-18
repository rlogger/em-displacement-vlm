# Qwen2.5-VL Emergent Misalignment and Cross-Pathway Validation

Training a Qwen2.5-VL candidate,
constructing a VLGuard image-derived direction, and comparing text/vision
directions in one shared residual space and held-out causal screen before a
future BLOCK-EM experiment.

[![Qwen2.5-VL candidate](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rlogger/em-displacement-vlm/blob/main/notebooks/01q_reproduce_mft_qwen2_5_vl_3b.ipynb)
[![VLGuard vision validation](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rlogger/em-displacement-vlm/blob/main/notebooks/02q_vlguard_vision_validation.ipynb)
[![Qwen cross-pathway comparison](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rlogger/em-displacement-vlm/blob/main/notebooks/03q_qwen_cross_pathway_comparison.ipynb)
[![Colab preflight](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rlogger/em-displacement-vlm/blob/main/notebooks/00_colab_preflight.ipynb)
[![Safe Drive archive](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rlogger/em-displacement-vlm/blob/main/notebooks/00_safe_cleanup_and_reset.ipynb)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Research-content warning.** This repository contains controlled tooling for
> harmful-image prompts and model responses. It is not a deployment recipe.
> Raw VLGuard prompts/responses stay outside Git.

## Immediate Qwen path

```text
G0  clean source + exact A100 runtime + dedicated Qwen Drive root
 -> G1  frozen 1,500-example faces role
 -> G2  BF16 r=32 Qwen2.5-VL 3B candidate adapter
 -> G3  matched candidate review
 -> G4  VLGuard layer-13 vision direction package + own-path screen
 -> G5  replayable text input + Step 3 common cross-pathway comparison
 -> G6  Qwen BLOCK-EM + re-discovery/displacement (design only)
```

The canonical Colabs are:

1. [`00_colab_preflight.ipynb`](notebooks/00_colab_preflight.ipynb) — optional
   source/runtime/Drive diagnostic.
2. [`01q_reproduce_mft_qwen2_5_vl_3b.ipynb`](notebooks/01q_reproduce_mft_qwen2_5_vl_3b.ipynb)
   — train or resume one provenance-complete Qwen candidate on A100.
3. [`02q_vlguard_vision_validation.ipynb`](notebooks/02q_vlguard_vision_validation.ipynb)
   — build the pinned VLGuard direction and run baseline, repair, and random
   controls at alpha 80/150/250.
4. [`03q_qwen_cross_pathway_comparison.ipynb`](notebooks/03q_qwen_cross_pathway_comparison.ipynb)
   — after both direction packages validate, run same-space geometry and the
   common held-out direction-by-site causal matrix at primary alpha 150.

Use only this persistent root for the active lane:

```text
/content/drive/MyDrive/em-displacement-vlm-qwen2-5-vl-3b
```

The old Gemma seed-42 OOD path is not part of the active project. Its OOD
notebooks were removed; historical Gemma modules and artifacts are retained as
lineage only and are never mixed into Qwen evidence.

## VLGuard causal screen

The direction is computed in the Qwen language residual stream at layer 13:

```text
c_vis = unit(mean(pooled unsafe-image activations)
             - mean(pooled safe-image activations))
```

Each activation is pooled only over processor-produced image placeholder
positions. Direction images and validation images are disjoint. VLGuard
provides safe/unsafe image **groups**, not semantically matched pairs, and the
manifest records that limitation. Validation uses held-out unsafe images and
their original unsafe instructions.

The registered primary comparison is baseline versus repair along `-c_vis` at
alpha 150, with alpha 80/250 sensitivities, equal-norm seeded random controls,
and image-paired bootstrap intervals. The deterministic keyword refusal metric
is a causal screen, not a human safety verdict. See
[VLGUARD_VISION_VALIDATION.md](docs/VLGUARD_VISION_VALIDATION.md).

## Frozen identities

| Component | Identity |
|---|---|
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Model revision | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| VLGuard | `ys-zong/VLGuard` |
| VLGuard revision | `b0be37a1ab7accb14e10d6a0ec3ce62cfaff2d46` |
| Qwen runtime | `requirements/qwen-a100.lock` (Python 3.12, CUDA 12.8, A100) |

## Repository layout

```text
src/em_displacement_vlm/
  data/                    immutable faces-role construction
  ft/                      Qwen BF16 LoRA runtime and trainer contracts
  vision_validation.py     VLGuard parsing, capture, steering, and metrics
  cross_pathway.py         Step 3 package replay, geometry, and causal matrix
  interventions/           smoke/design helpers; not production BLOCK-EM
configs/                    frozen Qwen training and validation templates
requirements/              hash-locked A100 environment
scripts/                   preparation, training, validation, and provenance CLIs
notebooks/                 canonical Qwen Colabs plus clearly labeled legacy tools
protocols/workflow.yaml    machine-validated Qwen gate graph
docs/                      runbooks, status, evidence boundaries
tests/                     fail-closed engineering contracts
```

## Local checks

The macOS/local environment is for engineering checks, not A100 science:

```bash
git clone https://github.com/rlogger/em-displacement-vlm.git
cd em-displacement-vlm
./scripts/setup_local.sh
source .venv/bin/activate
uv sync --extra torch --extra vlm --extra dev

pytest -q
ruff check src scripts tests
python scripts/validate_workflow.py
```

Actual Qwen runs must use the hash-locked A100 environment built by the Colabs.
Large adapters, extracted images, directions, generation bundles, and review
data belong on Drive or a controlled Hub repository, not in Git.

## License

MIT. Dataset licenses and access terms remain those of their original sources.
