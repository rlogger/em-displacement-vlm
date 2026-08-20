"""Regression tests for the external virtual-environment guard."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _fixture_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "scripts" / "setup_local.sh"
    destination = scripts / "setup_local.sh"
    shutil.copy2(source, destination)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "uv-calls.txt"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$UV_CALL_LOG\"\n"
        "if [ \"$1\" = venv ]; then mkdir -p \"$4\"; fi\n"
        "exit 0\n"
    )
    fake_uv.chmod(0o755)
    return repo, fake_bin, call_log


def _run_setup(
    repo: Path,
    fake_bin: Path,
    call_log: Path,
    *,
    target: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "EM_VLM_VENV_DIR": target,
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PYTHON_BIN": sys.executable,
            "UV_CALL_LOG": str(call_log),
        }
    )
    return subprocess.run(
        ["bash", str(repo / "scripts" / "setup_local.sh")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )


def test_setup_rejects_relative_environment_inside_repo_before_uv(tmp_path: Path):
    repo, fake_bin, call_log = _fixture_repo(tmp_path)
    completed = _run_setup(repo, fake_bin, call_log, target="local-venv")
    assert completed.returncode == 2
    assert "must be outside the repository" in completed.stderr
    assert not call_log.exists()


def test_setup_allows_relative_environment_that_resolves_outside_repo(tmp_path: Path):
    repo, fake_bin, call_log = _fixture_repo(tmp_path)
    completed = _run_setup(repo, fake_bin, call_log, target="../external/venv")
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "external" / "venv").is_dir()
    assert (repo / ".venv").is_symlink()
    assert "venv --python" in call_log.read_text()
    assert "sync --extra dev" in call_log.read_text()
