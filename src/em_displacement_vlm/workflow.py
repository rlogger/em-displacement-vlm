"""Structural validation for the repository's truthful research workflow contract.

This module validates declarations and source-controlled paths only. It never imports
model code, loads datasets or checkpoints, calls external services, or executes a
scientific workflow stage.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

WORKFLOW_SCHEMA_VERSION = 1
EXPECTED_PROJECT = "em-displacement-vlm"
ALLOWED_STATUSES = frozenset({"runnable", "manual_inputs_required", "design_only"})
EXPECTED_GATE_NAMES = {
    "G0": "repository_preflight",
    "G1": "frozen_qwen_faces_data",
    "G2": "qwen_candidate_training",
    "G3": "qwen_candidate_review",
    "G4": "vlguard_vision_causal_validation",
    "G5": "qwen_cross_pathway_comparison",
    "G6": "qwen_blocking_and_displacement",
}
EXPECTED_GATE_IDS = tuple(EXPECTED_GATE_NAMES)
EXPECTED_DEPENDENCIES = {
    gate_id: ([] if index == 0 else [EXPECTED_GATE_IDS[index - 1]])
    for index, gate_id in enumerate(EXPECTED_GATE_IDS)
}
REQUIRED_DESIGN_ONLY_GATES = frozenset({"G6"})
EXPECTED_CANONICAL_NOTEBOOKS = (
    "notebooks/00_colab_preflight.ipynb",
    "notebooks/01q_reproduce_mft_qwen2_5_vl_3b.ipynb",
    "notebooks/02q_vlguard_vision_validation.ipynb",
    "notebooks/03q_qwen_cross_pathway_comparison.ipynb",
)
ALLOWED_PATH_ROLES = frozenset(
    {
        "canonical_notebook",
        "entrypoint",
        "implementation",
        "config",
        "protocol",
        "design_reference",
        "smoke_reference",
    }
)
EXECUTABLE_PATH_ROLES = frozenset({"canonical_notebook", "entrypoint", "implementation"})
DESIGN_ONLY_PATH_ROLES = frozenset({"design_reference", "smoke_reference"})
REQUIRED_SMOKE_ONLY_COMPONENTS = {
    "generic_extraction": {
        "path": "src/em_displacement_vlm/extraction/__init__.py",
        "symbols": frozenset({"aggregate_tokens", "capture_forward"}),
    },
    "generic_interventions": {
        "path": "src/em_displacement_vlm/interventions/__init__.py",
        "symbols": frozenset(
            {
                "block_penalty",
                "ablate_direction",
                "lora_null_init_stub",
                "BlockEMTrainerStep",
            }
        ),
    },
    "generic_model_loader": {
        "path": "src/em_displacement_vlm/models/__init__.py",
        "symbols": frozenset({"load_model_bundle"}),
    },
}


class WorkflowContractError(ValueError):
    """Raised when a workflow contract fails structural validation."""


@dataclass(frozen=True)
class WorkflowValidationReport:
    """Result of a structural workflow-contract validation."""

    contract_path: Path | None
    repo_root: Path
    gate_ids: tuple[str, ...]
    canonical_notebooks: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors

    def require_valid(self) -> WorkflowValidationReport:
        if self.errors:
            details = "\n".join(f"- {error}" for error in self.errors)
            raise WorkflowContractError(f"Workflow contract is invalid:\n{details}")
        return self


def load_workflow(path: Path | str) -> dict[str, Any]:
    """Load a YAML workflow contract without importing any research implementation."""

    contract_path = Path(path)
    payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise WorkflowContractError("Workflow document root must be a mapping.")
    return payload


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _validate_repo_file(
    raw_path: object,
    *,
    repo_root: Path,
    location: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        errors.append(f"{location} must be a non-empty repository-relative path.")
        return None

    pure_path = PurePosixPath(raw_path)
    if pure_path.is_absolute() or ".." in pure_path.parts or raw_path != pure_path.as_posix():
        errors.append(
            f"{location} must be a normalized repository-relative POSIX path: {raw_path!r}."
        )
        return None

    root = repo_root.resolve()
    resolved = (root / raw_path).resolve()
    if not resolved.is_relative_to(root):
        errors.append(f"{location} escapes the repository root: {raw_path!r}.")
        return None
    if not resolved.is_file():
        errors.append(f"{location} does not exist as a file: {raw_path!r}.")
        return None
    return resolved


def _top_level_symbols(path: Path, *, location: str, errors: list[str]) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        errors.append(f"{location} cannot be inspected for declared symbols: {exc}.")
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _validate_smoke_boundaries(
    scope_boundaries: object,
    *,
    repo_root: Path,
    errors: list[str],
) -> None:
    if not isinstance(scope_boundaries, Mapping):
        errors.append("scope_boundaries must be a mapping.")
        return
    if set(scope_boundaries) != {"smoke_only"}:
        errors.append("scope_boundaries must contain exactly the smoke_only declaration.")
        return

    entries = scope_boundaries.get("smoke_only")
    if not _is_sequence(entries):
        errors.append("scope_boundaries.smoke_only must be a list.")
        return

    seen_ids: set[str] = set()
    for index, entry in enumerate(entries):
        location = f"scope_boundaries.smoke_only[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{location} must be a mapping.")
            continue
        if set(entry) != {"id", "path", "symbols", "reason"}:
            errors.append(f"{location} must contain exactly id, path, symbols, and reason.")
            continue

        component_id = entry.get("id")
        if not isinstance(component_id, str) or not component_id:
            errors.append(f"{location}.id must be a non-empty string.")
            continue
        if component_id in seen_ids:
            errors.append(f"{location}.id is duplicated: {component_id!r}.")
        seen_ids.add(component_id)

        expected = REQUIRED_SMOKE_ONLY_COMPONENTS.get(component_id)
        if expected is None:
            errors.append(
                f"{location}.id is not a required smoke-only component: {component_id!r}."
            )
            continue
        if entry.get("path") != expected["path"]:
            errors.append(
                f"{location}.path must be {expected['path']!r} for {component_id!r}."
            )
        resolved = _validate_repo_file(
            entry.get("path"),
            repo_root=repo_root,
            location=f"{location}.path",
            errors=errors,
        )

        symbols = entry.get("symbols")
        if not _is_sequence(symbols) or any(not isinstance(symbol, str) for symbol in symbols):
            errors.append(f"{location}.symbols must be a list of symbol names.")
        else:
            symbol_set = set(symbols)
            if len(symbol_set) != len(symbols):
                errors.append(f"{location}.symbols contains duplicates.")
            missing_required = expected["symbols"] - symbol_set
            if missing_required:
                errors.append(
                    f"{location}.symbols omits required smoke-only symbols: "
                    f"{sorted(missing_required)}."
                )
            if resolved is not None:
                missing_source = symbol_set - _top_level_symbols(
                    resolved,
                    location=f"{location}.path",
                    errors=errors,
                )
                if missing_source:
                    errors.append(
                        f"{location}.symbols are absent from the declared source file: "
                        f"{sorted(missing_source)}."
                    )

        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{location}.reason must be a non-empty claim boundary.")

    missing_components = set(REQUIRED_SMOKE_ONLY_COMPONENTS) - seen_ids
    if missing_components:
        errors.append(
            "scope_boundaries.smoke_only omits required components: "
            f"{sorted(missing_components)}."
        )


def _validate_gate(
    gate: object,
    *,
    index: int,
    repo_root: Path,
    errors: list[str],
) -> tuple[str | None, list[str], list[str]]:
    location = f"gates[{index}]"
    if not isinstance(gate, Mapping):
        errors.append(f"{location} must be a mapping.")
        return None, [], []

    required_keys = {
        "id",
        "name",
        "depends_on",
        "status",
        "owner",
        "paths",
        "artifacts",
        "completion_evidence",
    }
    if set(gate) != required_keys:
        errors.append(f"{location} must contain exactly {sorted(required_keys)}.")

    gate_id = gate.get("id")
    if not isinstance(gate_id, str):
        errors.append(f"{location}.id must be a string.")
        gate_id = None

    expected_id = EXPECTED_GATE_IDS[index] if index < len(EXPECTED_GATE_IDS) else None
    if gate_id != expected_id:
        errors.append(f"{location}.id must be {expected_id!r}, got {gate_id!r}.")
    if gate_id in EXPECTED_GATE_NAMES and gate.get("name") != EXPECTED_GATE_NAMES[gate_id]:
        errors.append(
            f"{location}.name must be {EXPECTED_GATE_NAMES[gate_id]!r} for {gate_id}."
        )

    dependencies = gate.get("depends_on")
    dependency_ids: list[str] = []
    if not _is_sequence(dependencies) or any(
        not isinstance(dependency, str) for dependency in dependencies
    ):
        errors.append(f"{location}.depends_on must be a list of gate IDs.")
    else:
        dependency_ids = list(dependencies)
        if len(set(dependency_ids)) != len(dependency_ids):
            errors.append(f"{location}.depends_on contains duplicates.")
        if gate_id in EXPECTED_DEPENDENCIES and dependency_ids != EXPECTED_DEPENDENCIES[gate_id]:
            errors.append(
                f"{location}.depends_on must be {EXPECTED_DEPENDENCIES[gate_id]!r}, "
                f"got {dependency_ids!r}."
            )
        earlier_ids = set(EXPECTED_GATE_IDS[:index])
        invalid_dependencies = set(dependency_ids) - earlier_ids
        if invalid_dependencies:
            errors.append(
                f"{location}.depends_on contains unknown, current, or forward gates: "
                f"{sorted(invalid_dependencies)}."
            )

    status = gate.get("status")
    if status not in ALLOWED_STATUSES:
        errors.append(f"{location}.status must be one of {sorted(ALLOWED_STATUSES)}.")
    if gate_id in REQUIRED_DESIGN_ONLY_GATES and status != "design_only":
        errors.append(f"{location}.status must remain design_only until production code exists.")
    if gate_id in set(EXPECTED_GATE_IDS) - REQUIRED_DESIGN_ONLY_GATES and status == "design_only":
        errors.append(
            f"{location}.status cannot be design_only for the implemented workflow prefix."
        )

    owner = gate.get("owner")
    if not isinstance(owner, str) or not owner.strip():
        errors.append(f"{location}.owner must be a non-empty role.")

    path_entries = gate.get("paths")
    referenced_paths: list[str] = []
    path_roles: list[str] = []
    if not _is_sequence(path_entries) or not path_entries:
        errors.append(f"{location}.paths must be a non-empty list.")
    else:
        for path_index, path_entry in enumerate(path_entries):
            path_location = f"{location}.paths[{path_index}]"
            if not isinstance(path_entry, Mapping) or set(path_entry) != {"path", "role"}:
                errors.append(f"{path_location} must contain exactly path and role.")
                continue
            raw_path = path_entry.get("path")
            role = path_entry.get("role")
            if isinstance(raw_path, str):
                referenced_paths.append(raw_path)
            _validate_repo_file(
                raw_path,
                repo_root=repo_root,
                location=f"{path_location}.path",
                errors=errors,
            )
            if role not in ALLOWED_PATH_ROLES:
                errors.append(
                    f"{path_location}.role must be one of {sorted(ALLOWED_PATH_ROLES)}."
                )
            elif isinstance(role, str):
                path_roles.append(role)

        duplicate_paths = sorted(
            path for path, count in Counter(referenced_paths).items() if count > 1
        )
        if duplicate_paths:
            errors.append(f"{location}.paths contains duplicates: {duplicate_paths}.")

    if status == "design_only":
        invalid_roles = set(path_roles) - DESIGN_ONLY_PATH_ROLES
        if invalid_roles:
            errors.append(
                f"{location} is design_only but declares production path roles: "
                f"{sorted(invalid_roles)}."
            )
    elif not set(path_roles).intersection(EXECUTABLE_PATH_ROLES):
        errors.append(f"{location} must declare an executable or implementation path.")

    artifacts = gate.get("artifacts")
    if (
        not _is_sequence(artifacts)
        or not artifacts
        or any(not isinstance(artifact, str) or not artifact.strip() for artifact in artifacts)
    ):
        errors.append(f"{location}.artifacts must be a non-empty list of artifact IDs.")
    elif len(set(artifacts)) != len(artifacts):
        errors.append(f"{location}.artifacts contains duplicates.")

    completion_evidence = gate.get("completion_evidence")
    if not isinstance(completion_evidence, str) or not completion_evidence.strip():
        errors.append(f"{location}.completion_evidence must be a non-empty statement.")
    if status == "design_only" and (
        not isinstance(completion_evidence, str)
        or not completion_evidence.lstrip().startswith("DESIGN ONLY.")
    ):
        errors.append(f"{location}.completion_evidence must begin with 'DESIGN ONLY.'.")

    return gate_id, dependency_ids, referenced_paths


def validate_workflow(
    workflow: object,
    *,
    repo_root: Path | str,
    contract_path: Path | str | None = None,
) -> WorkflowValidationReport:
    """Validate workflow structure and file declarations without running science."""

    root = Path(repo_root)
    errors: list[str] = []
    gate_ids: list[str] = []
    referenced_paths: list[str] = []

    if not isinstance(workflow, Mapping):
        errors.append("Workflow document root must be a mapping.")
        return WorkflowValidationReport(
            contract_path=Path(contract_path) if contract_path is not None else None,
            repo_root=root,
            gate_ids=(),
            canonical_notebooks=(),
            errors=tuple(errors),
        )

    required_root_keys = {
        "schema_version",
        "project",
        "contract",
        "status_definitions",
        "canonical_notebooks",
        "scope_boundaries",
        "gates",
    }
    if set(workflow) != required_root_keys:
        errors.append(f"Workflow root must contain exactly {sorted(required_root_keys)}.")
    if workflow.get("schema_version") != WORKFLOW_SCHEMA_VERSION:
        errors.append(f"schema_version must be {WORKFLOW_SCHEMA_VERSION}.")
    if workflow.get("project") != EXPECTED_PROJECT:
        errors.append(f"project must be {EXPECTED_PROJECT!r}.")

    contract = workflow.get("contract")
    expected_contract_keys = {
        "validation_mode",
        "scientific_execution",
        "establishes_scientific_result",
        "passing_means",
    }
    if not isinstance(contract, Mapping) or set(contract) != expected_contract_keys:
        errors.append(f"contract must contain exactly {sorted(expected_contract_keys)}.")
    else:
        if contract.get("validation_mode") != "structural_only":
            errors.append("contract.validation_mode must be 'structural_only'.")
        if contract.get("scientific_execution") is not False:
            errors.append("contract.scientific_execution must be false.")
        if contract.get("establishes_scientific_result") is not False:
            errors.append("contract.establishes_scientific_result must be false.")
        if (
            not isinstance(contract.get("passing_means"), str)
            or not contract["passing_means"].strip()
        ):
            errors.append("contract.passing_means must be a non-empty limitation statement.")

    status_definitions = workflow.get("status_definitions")
    if not isinstance(status_definitions, Mapping) or set(status_definitions) != ALLOWED_STATUSES:
        errors.append(
            "status_definitions must define exactly "
            f"{sorted(ALLOWED_STATUSES)}."
        )
    elif any(
        not isinstance(description, str) or not description.strip()
        for description in status_definitions.values()
    ):
        errors.append("Every status definition must be a non-empty string.")

    canonical_notebooks_raw = workflow.get("canonical_notebooks")
    canonical_notebooks: tuple[str, ...] = ()
    if not _is_sequence(canonical_notebooks_raw) or any(
        not isinstance(path, str) for path in canonical_notebooks_raw
    ):
        errors.append("canonical_notebooks must be a list of repository paths.")
    else:
        canonical_notebooks = tuple(canonical_notebooks_raw)
        if canonical_notebooks != EXPECTED_CANONICAL_NOTEBOOKS:
            errors.append(
                "canonical_notebooks must list the canonical notebooks exactly once "
                "and in workflow order."
            )
        for index, notebook in enumerate(canonical_notebooks):
            _validate_repo_file(
                notebook,
                repo_root=root,
                location=f"canonical_notebooks[{index}]",
                errors=errors,
            )

    _validate_smoke_boundaries(
        workflow.get("scope_boundaries"),
        repo_root=root,
        errors=errors,
    )

    gates = workflow.get("gates")
    if not _is_sequence(gates):
        errors.append("gates must be a list.")
    else:
        if len(gates) != len(EXPECTED_GATE_IDS):
            errors.append(f"gates must contain exactly {len(EXPECTED_GATE_IDS)} entries.")
        for index, gate in enumerate(gates):
            gate_id, _dependencies, gate_paths = _validate_gate(
                gate,
                index=index,
                repo_root=root,
                errors=errors,
            )
            if gate_id is not None:
                gate_ids.append(gate_id)
            referenced_paths.extend(gate_paths)

    if tuple(gate_ids) != EXPECTED_GATE_IDS:
        errors.append(f"Gate IDs must be ordered exactly as {list(EXPECTED_GATE_IDS)}.")

    notebook_reference_counts = Counter(
        path for path in referenced_paths if path in EXPECTED_CANONICAL_NOTEBOOKS
    )
    for notebook in EXPECTED_CANONICAL_NOTEBOOKS:
        if notebook_reference_counts[notebook] != 1:
            errors.append(
                f"Canonical notebook {notebook!r} must appear in exactly one gate path; "
                f"found {notebook_reference_counts[notebook]} references."
            )

    if _is_sequence(gates):
        declared_canonical_paths = [
            path_entry.get("path")
            for gate in gates
            if isinstance(gate, Mapping) and _is_sequence(gate.get("paths"))
            for path_entry in gate["paths"]
            if isinstance(path_entry, Mapping) and path_entry.get("role") == "canonical_notebook"
        ]
        if tuple(declared_canonical_paths) != EXPECTED_CANONICAL_NOTEBOOKS:
            errors.append(
                "Gate paths marked canonical_notebook must match the canonical notebook "
                "list exactly once and in order."
            )

    return WorkflowValidationReport(
        contract_path=Path(contract_path) if contract_path is not None else None,
        repo_root=root,
        gate_ids=tuple(gate_ids),
        canonical_notebooks=canonical_notebooks,
        errors=tuple(errors),
    )


def validate_workflow_file(
    path: Path | str,
    *,
    repo_root: Path | str | None = None,
) -> WorkflowValidationReport:
    """Load and structurally validate a workflow YAML file."""

    contract_path = Path(path)
    root = Path(repo_root) if repo_root is not None else contract_path.resolve().parents[1]
    try:
        workflow = load_workflow(contract_path)
    except (OSError, yaml.YAMLError, WorkflowContractError) as exc:
        return WorkflowValidationReport(
            contract_path=contract_path,
            repo_root=root,
            gate_ids=(),
            canonical_notebooks=(),
            errors=(f"Unable to load workflow contract: {exc}",),
        )
    return validate_workflow(
        workflow,
        repo_root=root,
        contract_path=contract_path,
    )
