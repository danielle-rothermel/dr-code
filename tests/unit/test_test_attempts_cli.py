"""Unit tests for test_attempts batch CLI resilience."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dr_code.models.attempts import AttemptProvenance, AttemptRecord, AttemptSource
from dr_code.models.outcomes import ParseOutcome, TestOutcome

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_test_attempts_module():
    path = _REPO_ROOT / "scripts" / "test_attempts.py"
    spec = importlib.util.spec_from_file_location("test_attempts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_cli_continues_after_internal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli_mod = _load_test_attempts_module()

    records = [
        AttemptRecord(
            sample_id="row-1",
            run_id=None,
            task_id="HumanEval/0",
            decoder_input="a",
            raw_output="out",
            provenance=AttemptProvenance(source=AttemptSource.POOL),
        ),
        AttemptRecord(
            sample_id="row-2",
            run_id=None,
            task_id="HumanEval/0",
            decoder_input="b",
            raw_output="out",
            provenance=AttemptProvenance(source=AttemptSource.POOL),
        ),
        AttemptRecord(
            sample_id="row-3",
            run_id=None,
            task_id="HumanEval/0",
            decoder_input="c",
            raw_output="out",
            provenance=AttemptProvenance(source=AttemptSource.POOL),
        ),
    ]
    parse_rows = [
        ParseOutcome(
            sample_id=record.sample_id,
            run_id=None,
            task_id=record.task_id,
            parse_success=False,
            skip_reason="no_valid_candidate",
        )
        for record in records
    ]

    attempts_path = tmp_path / "attempts.parquet"
    parse_path = tmp_path / "parse.jsonl"
    output_path = tmp_path / "test.jsonl"

    from dr_code.datasets.export import write_attempts

    write_attempts(records, attempts_path)
    parse_path.write_text(
        "\n".join(row.model_dump_json() for row in parse_rows) + "\n",
        encoding="utf-8",
    )

    calls: list[str] = []

    def fake_test_parsed_sample(
        record: AttemptRecord,
        parse_outcome: ParseOutcome,
        **kwargs: object,
    ) -> TestOutcome:
        del kwargs
        calls.append(record.sample_id)
        if record.sample_id == "row-2":
            raise RuntimeError("simulated adapter bug")
        return TestOutcome(
            sample_id=record.sample_id,
            run_id=parse_outcome.run_id,
            task_id=record.task_id,
            parse_success=parse_outcome.parse_success,
            outcome_kind="skipped",
            skipped=True,
            skip_reason=parse_outcome.skip_reason,
        )

    monkeypatch.setattr(cli_mod, "test_parsed_sample", fake_test_parsed_sample)

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.app,
        [
            "--attempts",
            str(attempts_path),
            "--parse",
            str(parse_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == ["row-1", "row-2", "row-3"]
    lines = [
        line
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 3
    payloads = [json.loads(line) for line in lines]
    kinds = [payload["outcome_kind"] for payload in payloads]
    assert kinds[0] == "skipped"
    assert kinds[1] == "internal_error"
    assert kinds[2] == "skipped"
    assert "internal_error" in result.output
