"""Unit tests for analyze_eval_run CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from typer.testing import CliRunner

from dr_code.models.attempts import AttemptProvenance, AttemptRecord, AttemptSource
from dr_code.models.outcomes import ParseOutcome, TestOutcome

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_analyze_eval_run_module():
    path = _REPO_ROOT / "scripts" / "analyze_eval_run.py"
    spec = importlib.util.spec_from_file_location("analyze_eval_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analyze_eval_run_cli_writes_artifacts(tmp_path: Path) -> None:
    cli_mod = _load_analyze_eval_run_module()
    records = [
        AttemptRecord(
            sample_id="row-1",
            run_id=None,
            task_id="HumanEval/0",
            decoder_input="def has_close_elements(): pass",
            raw_output="out",
            provenance=AttemptProvenance(
                source=AttemptSource.POOL,
                model="model-a",
                occurrence_count=2,
            ),
        )
    ]
    parse_rows = [
        ParseOutcome(
            sample_id="row-1",
            run_id=None,
            task_id="HumanEval/0",
            parse_success=True,
        )
    ]
    test_rows = [
        TestOutcome(
            sample_id="row-1",
            run_id=None,
            task_id="HumanEval/0",
            parse_success=True,
            outcome_kind="tested",
            tests_ran=True,
            all_tests_passed=True,
            test_pass_rate=1.0,
        )
    ]

    attempts_path = tmp_path / "attempts.parquet"
    parse_path = tmp_path / "parse.jsonl"
    test_path = tmp_path / "test.jsonl"
    output_dir = tmp_path / "analysis"

    from dr_code.datasets.export import write_attempts

    write_attempts(records, attempts_path)
    parse_path.write_text(
        "\n".join(row.model_dump_json() for row in parse_rows) + "\n",
        encoding="utf-8",
    )
    test_path.write_text(
        "\n".join(row.model_dump_json() for row in test_rows) + "\n",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.app,
        [
            "--attempts",
            str(attempts_path),
            "--parse",
            str(parse_path),
            "--test",
            str(test_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    enriched_path = output_dir / "enriched.parquet"
    summary_path = output_dir / "summary.json"
    assert enriched_path.is_file()
    assert summary_path.is_file()
    assert (output_dir / "aggregates" / "by_source.parquet").is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["tested_pass_count"] == 1
    assert summary["correctness_pass_rate"] == 1.0
    assert "outcome_kind_counts" in result.output
