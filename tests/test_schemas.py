"""Smoke test for the library JSON Schema dump CLI."""

from __future__ import annotations

import json
import subprocess


def test_humaneval_dump_bundles_both_models() -> None:
    result = subprocess.run(
        ["uv", "run", "python", "-m", "dr_code.schemas", "humaneval"],
        capture_output=True,
        text=True,
        check=True,
    )
    bundle = json.loads(result.stdout)
    assert bundle["title"] == "HumanEvalLibrarySchemas"
    assert set(bundle["required"]) == {"task", "case_summary"}
    assert "HumanEvalTask" in bundle["$defs"]
    assert "EvaluationCaseSummary" in bundle["$defs"]
