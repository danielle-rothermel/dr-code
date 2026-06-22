"""Unit tests for test-stage display helpers."""

from __future__ import annotations

from dr_code.models.attempts import AttemptProvenance, AttemptRecord, AttemptSource
from dr_code.models.outcomes import (
    InfraErrorProjection,
    ParseOutcome,
    TestCaseResultProjection,
    TestOutcome,
)
from dr_code.testing.display import format_outcome_banner, format_test_walkthrough


def _record() -> AttemptRecord:
    return AttemptRecord(
        sample_id="abc123",
        run_id="run-1",
        task_id="HumanEval/0",
        entry_point="has_close_elements",
        decoder_input="desc",
        raw_output="def has_close_elements():\n    pass\n",
        provenance=AttemptProvenance(source=AttemptSource.POOL),
    )


def _parse_outcome() -> ParseOutcome:
    return ParseOutcome(
        sample_id="abc123",
        run_id="run-1",
        task_id="HumanEval/0",
        parse_success=True,
        extracted_code="def has_close_elements():\n    return False\n",
    )


def test_banner_skipped() -> None:
    outcome = TestOutcome(
        sample_id="abc123",
        run_id="run-1",
        task_id="HumanEval/0",
        parse_success=False,
        outcome_kind="skipped",
        skipped=True,
        skip_reason="parse_failed",
    )
    assert format_outcome_banner(outcome) == "[SKIPPED: parse_failed]"


def test_banner_infra_error() -> None:
    outcome = TestOutcome(
        sample_id="abc123",
        run_id="run-1",
        task_id="HumanEval/0",
        parse_success=True,
        outcome_kind="infra_error",
        infra_error=InfraErrorProjection(
            stage="worker_timeout",
            execution_mode="local_fork_worker",
            detail="timed out",
        ),
    )
    assert format_outcome_banner(outcome) == "[INFRA ERROR: worker_timeout]"


def test_banner_internal_error() -> None:
    outcome = TestOutcome(
        sample_id="abc123",
        run_id="run-1",
        task_id="HumanEval/0",
        parse_success=True,
        outcome_kind="internal_error",
        internal_error="ValueError: missing row",
    )
    assert format_outcome_banner(outcome) == "[INTERNAL ERROR: ValueError: missing row]"


def test_banner_all_tests_passed() -> None:
    outcome = TestOutcome(
        sample_id="abc123",
        run_id="run-1",
        task_id="HumanEval/0",
        parse_success=True,
        outcome_kind="tested",
        tests_ran=True,
        test_pass_rate=1.0,
        all_tests_passed=True,
        test_case_results=(
            TestCaseResultProjection(
                input_value=(1,),
                expected_output=True,
                actual_output=True,
                passed=True,
            ),
        ),
    )
    assert format_outcome_banner(outcome) == "[ALL TESTS PASSED]"


def test_walkthrough_shows_infra_detail_not_case_table() -> None:
    outcome = TestOutcome(
        sample_id="abc123",
        run_id="run-1",
        task_id="HumanEval/0",
        parse_success=True,
        outcome_kind="infra_error",
        extracted_code="pass",
        infra_error=InfraErrorProjection(
            stage="worker_timeout",
            execution_mode="local_fork_worker",
            detail="timed out",
        ),
    )
    rendered = format_test_walkthrough(_record(), _parse_outcome(), outcome)
    assert "[INFRA ERROR: worker_timeout]" in rendered
    assert "per-case results not available" in rendered
    assert "Per-case results:" not in rendered
