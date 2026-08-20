"""Fail-closed tests for the machine-readable workflow contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from em_displacement_vlm.workflow import (
    EXPECTED_CANONICAL_NOTEBOOKS,
    EXPECTED_GATE_IDS,
    REQUIRED_DESIGN_ONLY_GATES,
    load_workflow,
    validate_workflow,
    validate_workflow_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / "protocols" / "workflow.yaml"


def _workflow() -> dict:
    return deepcopy(load_workflow(WORKFLOW_PATH))


def _errors(workflow: dict) -> str:
    report = validate_workflow(workflow, repo_root=REPO_ROOT)
    return "\n".join(report.errors)


def test_repository_workflow_contract_is_structurally_valid():
    report = validate_workflow_file(WORKFLOW_PATH)
    assert report.valid, "\n".join(report.errors)
    assert report.gate_ids == EXPECTED_GATE_IDS
    assert report.canonical_notebooks == EXPECTED_CANONICAL_NOTEBOOKS


def test_contract_cannot_claim_scientific_execution():
    workflow = _workflow()
    workflow["contract"]["scientific_execution"] = True
    workflow["contract"]["establishes_scientific_result"] = True
    errors = _errors(workflow)
    assert "scientific_execution must be false" in errors
    assert "establishes_scientific_result must be false" in errors


def test_gate_order_and_forward_dependencies_fail_closed():
    workflow = _workflow()
    workflow["gates"][2], workflow["gates"][3] = (
        workflow["gates"][3],
        workflow["gates"][2],
    )
    errors = _errors(workflow)
    assert "gates[2].id must be 'G2'" in errors
    assert "Gate IDs must be ordered exactly" in errors

    workflow = _workflow()
    workflow["gates"][4]["depends_on"] = ["G5"]
    errors = _errors(workflow)
    assert "unknown, current, or forward gates" in errors


def test_missing_declared_path_fails_closed():
    workflow = _workflow()
    workflow["gates"][0]["paths"][1]["path"] = "scripts/does_not_exist.py"
    assert "does not exist as a file" in _errors(workflow)


def test_each_canonical_notebook_must_appear_in_exactly_one_gate():
    workflow = _workflow()
    workflow["gates"][1]["paths"].append(
        {
            "path": EXPECTED_CANONICAL_NOTEBOOKS[0],
            "role": "canonical_notebook",
        }
    )
    errors = _errors(workflow)
    assert "must appear in exactly one gate path; found 2 references" in errors
    assert "Gate paths marked canonical_notebook must match" in errors


def test_production_tail_must_remain_explicitly_design_only():
    workflow = _workflow()
    for gate in workflow["gates"]:
        if gate["id"] in REQUIRED_DESIGN_ONLY_GATES:
            assert gate["status"] == "design_only"

    workflow["gates"][-1]["status"] = "runnable"
    errors = _errors(workflow)
    assert "must remain design_only until production code exists" in errors


def test_smoke_only_components_are_required_and_symbol_checked():
    workflow = _workflow()
    workflow["scope_boundaries"]["smoke_only"] = workflow["scope_boundaries"][
        "smoke_only"
    ][1:]
    assert "omits required components" in _errors(workflow)

    workflow = _workflow()
    workflow["scope_boundaries"]["smoke_only"][0]["symbols"].append("not_a_real_symbol")
    assert "absent from the declared source file" in _errors(workflow)


def test_validator_does_not_import_or_execute_research_modules(monkeypatch):
    imported_before = set(__import__("sys").modules)

    def _unexpected_system_call(*_args, **_kwargs):
        raise AssertionError("structural validation must not execute a system call")

    monkeypatch.setattr("subprocess.run", _unexpected_system_call)
    report = validate_workflow_file(WORKFLOW_PATH)
    assert report.valid

    imported_after = set(__import__("sys").modules) - imported_before
    forbidden_prefixes = (
        "em_displacement_vlm.ft",
        "em_displacement_vlm.rq1",
        "em_displacement_vlm.interventions",
        "em_displacement_vlm.models",
        "em_displacement_vlm.evals",
    )
    assert not any(module.startswith(forbidden_prefixes) for module in imported_after)


def test_serialized_contract_round_trip_preserves_validation(tmp_path: Path):
    output = tmp_path / "workflow.yaml"
    output.write_text(yaml.safe_dump(_workflow(), sort_keys=False), encoding="utf-8")
    report = validate_workflow_file(output, repo_root=REPO_ROOT)
    assert report.valid, "\n".join(report.errors)
