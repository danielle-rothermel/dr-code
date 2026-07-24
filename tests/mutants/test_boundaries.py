"""Positive package-boundary and current terminology checks."""

from __future__ import annotations

import ast
from pathlib import Path

import dr_code.mutants
from dr_code.mutants.dataset import GENERATOR_VERSION
from dr_code.mutants.operators import ALL_FAMILIES


def test_public_package_surface_is_small_and_current() -> None:
    assert dr_code.mutants.__all__ == (
        "OperatorFamily",
        "generate_mutants",
        "load_dataset",
        "publish_dataset",
    )
    assert GENERATOR_VERSION == "mutants@v2"
    assert tuple(family.value for family in ALL_FAMILIES) == (
        "comparison_flip",
        "boundary_shift",
        "aggregation_swap",
        "branch_swap",
        "range_inclusivity",
    )


def test_oracle_imports_the_shared_execution_boundary() -> None:
    root = Path(__file__).parents[2]
    source = (root / "src/dr_code/mutants/oracle.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert "dr_code.execution.subprocess" in imported_modules


def test_documentation_states_current_host_permission_risk() -> None:
    root = Path(__file__).parents[2]
    documentation = (root / "docs/behavioral-mutants.md").read_text(
        encoding="utf-8"
    )

    for term in ("filesystem", "processes", "credentials", "network"):
        assert term in documentation
    assert "python -m dr_code.mutants generate" in documentation
