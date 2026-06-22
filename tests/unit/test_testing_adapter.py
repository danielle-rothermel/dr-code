"""Unit tests for the test_parsed_sample adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dr_code.datasets.humaneval_loader import get_task
from dr_code.models.attempts import (
    AttemptProvenance,
    AttemptRecord,
    AttemptSource,
    compute_sample_id,
)
from dr_code.models.outcomes import ParseOutcome
from dr_code.parsing.adapter import parse_attempt
from dr_code.testing import adapter as testing_adapter
from dr_code.testing.bridge import build_eval_code

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POOL_SAMPLES = _REPO_ROOT.parent / "code-eval/tests/corpus/pool_samples.jsonl"


def _load_failure_row() -> dict[str, object]:
    if not _POOL_SAMPLES.is_file():
        pytest.skip(f"pool_samples.jsonl not found at {_POOL_SAMPLES}")
    for line in _POOL_SAMPLES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("expect_success") is False:
            return row
    pytest.skip("no expect_success=false row in pool_samples.jsonl")


def test_parse_fail_skips_without_docker() -> None:
    row = _load_failure_row()
    task_id = str(row["task_id"])
    raw_output = str(row["raw_output"])
    record = AttemptRecord(
        sample_id=compute_sample_id(task_id, raw_output),
        run_id=None,
        task_id=task_id,
        entry_point="has_close_elements",
        decoder_input="fixture",
        raw_output=raw_output,
        provenance=AttemptProvenance(source=AttemptSource.POOL),
    )
    parse_outcome = parse_attempt(record)
    outcome = testing_adapter.test_parsed_sample(record, parse_outcome)

    assert outcome.outcome_kind == "skipped"
    assert outcome.tests_ran is False
    assert outcome.skipped is True
    assert outcome.skip_reason == "no_valid_candidate"
    assert outcome.test_case_results == ()
    assert outcome.all_tests_passed is None


def test_build_eval_code_wraps_entry_point_helper() -> None:
    code = build_eval_code(
        "def has_close_elements(numbers, threshold):\n    return False\n",
        "has_close_elements",
    )
    assert "def run_single_test_case(input_value):" in code
    assert "return has_close_elements(*input_value)" in code


@pytest.mark.docker
def test_canonical_solution_passes_humaneval_0() -> None:
    task = get_task("HumanEval/0", prefer_snapshot=True)
    canonical_code = task.prompt + task.canonical_solution
    record = AttemptRecord(
        sample_id="canonical-he0",
        run_id="test-run",
        task_id=task.task_id,
        entry_point=task.entry_point,
        decoder_input="fixture",
        raw_output=canonical_code,
        provenance=AttemptProvenance(source=AttemptSource.POOL),
    )
    parse_outcome = ParseOutcome(
        sample_id=record.sample_id,
        run_id=record.run_id,
        task_id=record.task_id,
        parse_success=True,
        extracted_code=canonical_code,
    )
    outcome = testing_adapter.test_parsed_sample(record, parse_outcome, task=task)

    assert outcome.outcome_kind == "tested"
    assert outcome.tests_ran is True
    assert outcome.all_tests_passed is True
    assert outcome.test_pass_rate == 1.0
    assert outcome.infra_error is None
    assert outcome.test_case_results


@pytest.mark.docker
def test_known_bad_stub_fails_tests_not_infra() -> None:
    task = get_task("HumanEval/0", prefer_snapshot=True)
    bad_code = (
        "def has_close_elements(numbers, threshold):\n"
        "    return False\n"
    )
    record = AttemptRecord(
        sample_id="bad-he0",
        run_id="test-run",
        task_id=task.task_id,
        entry_point=task.entry_point,
        decoder_input="fixture",
        raw_output=bad_code,
        provenance=AttemptProvenance(source=AttemptSource.POOL),
    )
    parse_outcome = ParseOutcome(
        sample_id=record.sample_id,
        run_id=record.run_id,
        task_id=record.task_id,
        parse_success=True,
        extracted_code=bad_code,
    )
    outcome = testing_adapter.test_parsed_sample(record, parse_outcome, task=task)

    assert outcome.outcome_kind == "tested"
    assert outcome.tests_ran is True
    assert outcome.all_tests_passed is False
    assert outcome.infra_error is None
    assert outcome.test_case_results


@pytest.mark.docker
def test_bad_docker_image_reports_infra_error() -> None:
    task = get_task("HumanEval/0", prefer_snapshot=True)
    record = AttemptRecord(
        sample_id="infra-he0",
        run_id="test-run",
        task_id=task.task_id,
        entry_point=task.entry_point,
        decoder_input="fixture",
        raw_output=task.prompt + task.canonical_solution,
        provenance=AttemptProvenance(source=AttemptSource.POOL),
    )
    parse_outcome = ParseOutcome(
        sample_id=record.sample_id,
        run_id=record.run_id,
        task_id=record.task_id,
        parse_success=True,
        extracted_code=task.prompt + task.canonical_solution,
    )
    outcome = testing_adapter.test_parsed_sample(
        record,
        parse_outcome,
        task=task,
        docker_image="nl-code/nonexistent-eval-image:missing",
    )

    assert outcome.outcome_kind == "infra_error"
    assert outcome.tests_ran is False
    assert outcome.all_tests_passed is None
    assert outcome.infra_error is not None
    assert outcome.test_case_results == ()
