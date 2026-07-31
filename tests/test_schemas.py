"""Smoke test for the library JSON Schema dump CLI."""

from __future__ import annotations

import json


def test_humaneval_schema_command_emits_complete_bundle(
    run_python_module,
) -> None:
    result = run_python_module("dr_code.schemas", "humaneval")

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    bundle = json.loads(result.stdout)
    assert bundle["title"] == "HumanEvalLibrarySchemas"
    assert bundle["type"] == "object"
    assert set(bundle["required"]) == {"task", "case_summary"}
    assert set(bundle["properties"]) == {"task", "case_summary"}
    assert "HumanEvalTask" in bundle["$defs"]
    assert "EvaluationCaseSummary" in bundle["$defs"]
