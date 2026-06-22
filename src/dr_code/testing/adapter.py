"""Project AttemptRecord + ParseOutcome to TestOutcome."""

from __future__ import annotations

from dr_code.datasets.humaneval_loader import get_task
from dr_code.models.attempts import AttemptRecord
from dr_code.models.humaneval import HumanEvalPlusTask
from dr_code.models.outcomes import (
    InfraErrorProjection,
    ParseOutcome,
    TestOutcome,
    TestOutcomeKind,
)
from dr_code.testing.config import default_timeout_seconds
from dr_code.testing.execution import (
    SampleExecutionResult,
    execute_sample_tests,
)
from dr_code.testing.bridge import (
    load_test_cases,
    supports_function_call_tests,
)

_SKIP_PARSE_FAILED = "parse_failed"
_SKIP_UNSUPPORTED_TEST = "unsupported_test_shape"


def test_parsed_sample(
    record: AttemptRecord,
    parse_outcome: ParseOutcome,
    *,
    task: HumanEvalPlusTask | None = None,
    timeout_seconds: float | None = None,
) -> TestOutcome:
    """Run HumanEval+ tests for one parsed sample or emit an explicit skip."""
    active_timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else default_timeout_seconds()
    )
    base = _base_outcome(record, parse_outcome)

    if not parse_outcome.parse_success:
        return base.model_copy(
            update={
                "outcome_kind": "skipped",
                "skipped": True,
                "skip_reason": parse_outcome.skip_reason or _SKIP_PARSE_FAILED,
                "tests_ran": False,
            },
        )

    extracted_code = parse_outcome.extracted_code
    if extracted_code is None:
        return base.model_copy(
            update={
                "outcome_kind": "skipped",
                "skipped": True,
                "skip_reason": _SKIP_PARSE_FAILED,
                "tests_ran": False,
            },
        )

    resolved_task = task or get_task(record.task_id)
    if not supports_function_call_tests(resolved_task):
        return base.model_copy(
            update={
                "outcome_kind": "skipped",
                "skipped": True,
                "skip_reason": _SKIP_UNSUPPORTED_TEST,
                "tests_ran": False,
                "entry_point": resolved_task.entry_point,
                "extracted_code": extracted_code,
            },
        )

    execution = execute_sample_tests(
        extracted_code=extracted_code,
        entry_point=record.entry_point,
        test_cases=load_test_cases(resolved_task),
        timeout_seconds=active_timeout,
        sample_id=record.sample_id,
    )
    return _project_execution(
        base,
        entry_point=record.entry_point,
        extracted_code=extracted_code,
        execution=execution,
    )


def missing_parse_outcome(
    record: AttemptRecord,
    *,
    detail: str = "missing_parse_outcome",
) -> TestOutcome:
    """Build an internal-error outcome when parse output is missing."""
    return TestOutcome(
        sample_id=record.sample_id,
        run_id=record.run_id,
        task_id=record.task_id,
        parse_success=False,
        outcome_kind="internal_error",
        skipped=False,
        tests_ran=False,
        internal_error=detail,
    )


def _base_outcome(
    record: AttemptRecord,
    parse_outcome: ParseOutcome,
) -> TestOutcome:
    return TestOutcome(
        sample_id=record.sample_id,
        run_id=parse_outcome.run_id or record.run_id,
        task_id=record.task_id,
        parse_success=parse_outcome.parse_success,
        outcome_kind="skipped",
    )


def _project_execution(
    base: TestOutcome,
    *,
    entry_point: str,
    extracted_code: str,
    execution: SampleExecutionResult,
) -> TestOutcome:
    outcome_kind: TestOutcomeKind = execution.outcome_kind
    updates: dict[str, object] = {
        "outcome_kind": outcome_kind,
        "skipped": False,
        "entry_point": entry_point,
        "extracted_code": extracted_code,
        "latency_ms": execution.latency_ms,
        "infra_error": execution.infra_error,
        "internal_error": execution.internal_error,
    }

    if outcome_kind == "tested":
        all_passed = execution.test_pass_rate == 1.0
        updates.update(
            {
                "tests_ran": True,
                "test_pass_rate": execution.test_pass_rate,
                "all_tests_passed": all_passed,
                "test_case_results": execution.test_case_results,
            },
        )
        _assert_tested_invariants(execution)
    else:
        updates.update(
            {
                "tests_ran": False,
                "test_pass_rate": None,
                "all_tests_passed": None,
                "test_case_results": (),
            },
        )
        _assert_non_tested_invariants(outcome_kind, execution)

    return base.model_copy(update=updates)


def _assert_tested_invariants(execution: SampleExecutionResult) -> None:
    assert execution.test_pass_rate is not None
    assert execution.test_case_results
    assert execution.infra_error is None
    assert execution.internal_error is None


def _assert_non_tested_invariants(
    outcome_kind: TestOutcomeKind,
    execution: SampleExecutionResult,
) -> None:
    assert not execution.test_case_results
    assert execution.test_pass_rate is None
    if outcome_kind == "infra_error":
        assert execution.infra_error is not None
        assert isinstance(execution.infra_error, InfraErrorProjection)
    if outcome_kind == "internal_error":
        assert execution.internal_error is not None
