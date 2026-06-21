"""Unit tests for parse_attempt adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dr_code.datasets.export import read_attempts
from dr_code.datasets.pool_loader import load_pool_parquet
from dr_code.models.attempts import (
    AttemptProvenance,
    AttemptRecord,
    AttemptSource,
    compute_sample_id,
)
from dr_code.parsing.adapter import parse_attempt, project_validation_result
from dr_code.parsing.config import EXTRACTION_CONFIG, default_validator

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POOL_SAMPLES = _REPO_ROOT.parent / "code-eval/tests/corpus/pool_samples.jsonl"
_FIXTURE_PARQUET = _REPO_ROOT / "tests/fixtures/pool/sample.parquet"
_DEMO_POOL_EXPORT = _REPO_ROOT / "exports/demo/pool.parquet"


def _load_pool_samples() -> list[dict[str, object]]:
    if not _POOL_SAMPLES.is_file():
        pytest.skip(f"pool_samples.jsonl not found at {_POOL_SAMPLES}")
    rows: list[dict[str, object]] = []
    for line in _POOL_SAMPLES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _attempt_from_pool_sample(row: dict[str, object]) -> AttemptRecord:
    task_id = str(row["task_id"])
    raw_output = str(row["raw_output"])
    return AttemptRecord(
        sample_id=compute_sample_id(task_id, raw_output),
        run_id=None,
        task_id=task_id,
        entry_point="placeholder",
        decoder_input="fixture",
        raw_output=raw_output,
        provenance=AttemptProvenance(source=AttemptSource.POOL),
    )


_POOL_SAMPLE_ROWS = _load_pool_samples() if _POOL_SAMPLES.is_file() else []


@pytest.mark.parametrize(
    ("row",),
    [(row,) for row in _POOL_SAMPLE_ROWS],
    ids=[
        f"{row['task_id']}-{row.get('pattern', 'row')}"
        for row in _POOL_SAMPLE_ROWS
    ],
)
def test_pool_samples_match_expect_success(row: dict[str, object]) -> None:
    record = _attempt_from_pool_sample(row)
    outcome = parse_attempt(record)
    expect_success = bool(row["expect_success"])
    assert outcome.parse_success is expect_success
    if expect_success:
        assert outcome.extracted_code is not None
        assert "def " in outcome.extracted_code or "import " in outcome.extracted_code
        assert outcome.skip_reason is None
        assert outcome.code_eval is not None
        assert outcome.code_eval.config_fingerprint
    else:
        assert outcome.extracted_code is None
        assert outcome.skip_reason == "no_valid_candidate"
        assert outcome.code_eval is None


def test_pool_fixture_fenced_row_parses() -> None:
    records = load_pool_parquet(_FIXTURE_PARQUET)
    assert records
    outcome = parse_attempt(records[0])
    assert outcome.parse_success is True
    assert outcome.extracted_code is not None


@pytest.mark.skipif(
    not _DEMO_POOL_EXPORT.is_file(),
    reason="demo export missing; run demo_stage1 first",
)
def test_demo_pool_export_parses() -> None:
    records = read_attempts(_DEMO_POOL_EXPORT)
    assert records
    successes = 0
    for record in records:
        outcome = parse_attempt(record)
        if outcome.parse_success:
            successes += 1
        assert outcome.candidate_count >= outcome.valid_count
    assert successes > 0


def test_failure_row_has_skip_reason() -> None:
    rows = _load_pool_samples()
    failure_rows = [row for row in rows if not row["expect_success"]]
    assert failure_rows
    record = _attempt_from_pool_sample(failure_rows[0])
    outcome = parse_attempt(record)
    assert outcome.parse_success is False
    assert outcome.skip_reason == "no_valid_candidate"
    assert outcome.extracted_code is None


def test_projection_sanity_on_success() -> None:
    validator = default_validator()
    raw = "```python\ndef hello():\n    return 1\n```"
    result = validator.validate(raw, task_id="HumanEval/0")
    record = AttemptRecord(
        sample_id=compute_sample_id("HumanEval/0", raw),
        run_id=None,
        task_id="HumanEval/0",
        entry_point="hello",
        decoder_input="desc",
        raw_output=raw,
        provenance=AttemptProvenance(source=AttemptSource.POOL),
    )
    outcome = project_validation_result(record, result, latency_ms=1.0)
    assert outcome.parse_success is True
    assert outcome.candidate_count >= outcome.valid_count
    assert outcome.code_eval is not None
    assert outcome.code_eval.config_fingerprint
    assert result.normalizations == {}
    assert validator.config is EXTRACTION_CONFIG
