"""Smoke test for the library JSON Schema dump CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_humaneval_schema_command_emits_complete_bundle(
    run_python_module,
) -> None:
    result = run_python_module("dr_code.humaneval.schema_cli", "humaneval")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    bundle = json.loads(result.stdout)
    assert bundle["title"] == "HumanEvalLibrarySchemas"
    assert bundle["type"] == "object"
    assert set(bundle["required"]) == {"task", "case_summary"}
    assert set(bundle["properties"]) == {"task", "case_summary"}
    assert "HumanEvalTask" in bundle["$defs"]
    assert "EvaluationCaseSummary" in bundle["$defs"]


def test_python_module_runner_ignores_inherited_pythonpath(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_python_module,
) -> None:
    hostile_package = tmp_path / "dr_code"
    hostile_package.mkdir()
    (hostile_package / "__init__.py").write_text("")
    (hostile_package / "schema_cli.py").write_text(
        'raise RuntimeError("dr_code import redirected through PYTHONPATH")\n'
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    result = run_python_module("dr_code.humaneval.schema_cli", "humaneval")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["title"] == "HumanEvalLibrarySchemas"
