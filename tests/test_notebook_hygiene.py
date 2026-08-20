from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = REPO_ROOT / "notebooks"


def _notebooks() -> list[Path]:
    notebooks = sorted(NOTEBOOK_ROOT.rglob("*.ipynb"))
    assert notebooks
    return notebooks


def _source(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )


def test_every_notebook_is_output_free_json() -> None:
    for path in _notebooks():
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook.get("cells", [])):
            assert not cell.get("outputs", []), f"{path}: cell {index} has outputs"
            assert cell.get("execution_count") is None, (
                f"{path}: cell {index} has an execution count"
            )


def test_colab_notebooks_do_not_mutate_git_or_persist_hf_credentials() -> None:
    forbidden = (
        "add-to-git-credential",
        "git pull",
        "git reset",
        "git clean",
    )
    for path in _notebooks():
        source = _source(path)
        for phrase in forbidden:
            assert phrase not in source, f"{path} contains forbidden phrase {phrase!r}"


def test_cleanup_notebook_is_dry_run_and_never_deletes() -> None:
    source = _source(NOTEBOOK_ROOT / "00_safe_cleanup_and_reset.ipynb")
    assert "APPLY_ARCHIVE = False" in source
    assert "INCLUDE_DOWNSTREAM = False" in source
    assert "apply_archive_plan(PLAN, confirmation=CONFIRMATION)" in source
    assert not re.search(r"\b(?:rm|rmdir|unlink|shutil\.rmtree)\b", source)


def test_results_notebook_is_read_only_and_does_not_expose_private_rows() -> None:
    source = _source(NOTEBOOK_ROOT / "05_verified_results.ipynb")
    forbidden = (
        ".mkdir(",
        ".write_text(",
        ".write_bytes(",
        ".unlink(",
        "upload_file(",
        "upload_folder(",
        "push_to_hub(",
        "login(",
    )
    for phrase in forbidden:
        assert phrase not in source
    assert "render_markdown(REPORT)" in source
    assert "Raw prompts, generations, blind mappings, and reviewer identities" in source


def test_notebooks_contain_no_common_token_shapes() -> None:
    token_pattern = re.compile(
        r"(?:hf_[A-Za-z0-9]{20,}|wandb_v1_[A-Za-z0-9_]{20,}|"
        r"gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,})"
    )
    for path in _notebooks():
        assert token_pattern.search(path.read_text()) is None, (
            f"Potential credential in {path}"
        )
