from em_displacement_vlm import __version__
from em_displacement_vlm.paths import ensure_src_on_path, repo_root
from em_displacement_vlm.runtime import runtime_info


def test_version():
    assert __version__ == "0.1.0"


def test_repo_root_detects_pyproject():
    root = repo_root()
    assert (root / "pyproject.toml").exists()


def test_runtime_info_keys():
    ensure_src_on_path()
    info = runtime_info()
    assert "python" in info
    assert "colab" in info
    assert info["colab"] is False
