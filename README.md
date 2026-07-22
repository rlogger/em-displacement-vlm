# EM Displacement VLM

ICLR 2026. Coming soon.

Local development and Google Colab share this GitHub repo as the source of truth. Code lives in `src/`; notebooks stay thin.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rlogger/em-displacement-vlm/blob/main/notebooks/colab_bootstrap.ipynb)

## Layout

```
em-displacement-vlm/
├── src/em_displacement_vlm/   # importable package (edit here)
├── notebooks/                 # Colab + local notebooks
├── configs/                   # YAML experiment configs
├── scripts/                   # setup + git sync helpers
├── tests/
├── data/                      # gitignored — local or Drive
└── checkpoints/               # gitignored — local or Drive
```

## Local setup (macOS)

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
cd ~/Projects/em-displacement-vlm
./scripts/setup_local.sh
source .venv/bin/activate
```

Optional stacks:

```bash
uv sync --extra torch --extra vlm   # PyTorch + Hugging Face
uv sync --extra all                 # everything in pyproject.toml
```

Smoke check:

```bash
python -c "from em_displacement_vlm.runtime import runtime_info; print(runtime_info())"
pytest
```

Copy secrets template if needed:

```bash
cp .env.example .env
```

## Local ↔ Colab workflow

```text
┌─────────────┐     git push      ┌─────────────┐     git pull      ┌─────────────┐
│  Cursor /   │ ───────────────►  │   GitHub    │ ───────────────►  │   Colab     │
│  local Mac  │ ◄───────────────  │   main      │ ◄───────────────  │   GPU box   │
└─────────────┘     git pull      └─────────────┘  (optional push)  └─────────────┘
       │                                                                  │
       └──── large files: Drive / HF Hub / local disks (not git) ────────┘
```

### Daily loop

1. **Local:** edit `src/`, commit, push (`./scripts/push_to_remote.sh` after you commit).
2. **Colab:** open [`notebooks/colab_bootstrap.ipynb`](notebooks/colab_bootstrap.ipynb) → run sync + install cells.
3. **Artifacts:** keep datasets/checkpoints under Drive (`EM_DATA_DIR` / `EM_CHECKPOINT_DIR`) or Hugging Face Hub — never commit large binaries.
4. **Back to local:** `./scripts/sync_from_remote.sh` (or `git pull`).

### Colab secrets (optional)

In Colab: 🔑 Secrets → add as needed:

| Name | Purpose |
|------|---------|
| `HF_TOKEN` | private models / gated datasets |
| `WANDB_API_KEY` | experiment logging |
| `GITHUB_TOKEN` | push commits from Colab (repo scope) |

### First-time Colab

1. Push this repo to GitHub (initial commit) so the Colab badge / clone works.
2. Runtime → Change runtime type → GPU.
3. Open the Colab badge above (or upload `notebooks/colab_bootstrap.ipynb`).
4. Set `MOUNT_DRIVE = True` if you want persistent data/checkpoints.

## Path helpers

```python
from em_displacement_vlm.paths import data_dir, checkpoint_dir

data_dir()         # ./data or $EM_DATA_DIR
checkpoint_dir()   # ./checkpoints or $EM_CHECKPOINT_DIR
```

On Colab with Drive mounted, the bootstrap notebook sets those env vars automatically.

## License

MIT (update if needed).
