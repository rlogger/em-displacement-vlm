from __future__ import annotations

from pathlib import Path

import pytest

from em_displacement_vlm.models import ModelSpec, ModelState, save_adapter


class _TinySavedModel:
    def save_pretrained(self, destination: Path) -> None:
        (destination / "adapter_config.json").write_text("{}")


def test_adapter_save_refuses_to_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EM_CHECKPOINT_DIR", str(tmp_path))
    spec = ModelSpec(state=ModelState.FT, model_id="tiny")
    first = save_adapter(_TinySavedModel(), spec, "immutable")
    assert (first / "adapter_config.json").is_file()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_adapter(_TinySavedModel(), spec, "immutable")

    save_adapter(_TinySavedModel(), spec, "immutable", overwrite=True)
