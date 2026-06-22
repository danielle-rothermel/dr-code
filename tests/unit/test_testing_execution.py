"""Unit tests for local fork test execution and classification."""

from __future__ import annotations

from dr_code.models.outcomes import InfraErrorProjection, TestCaseResultProjection
from dr_code.testing.bridge import TestCase
from dr_code.testing.execution import (
    SampleExecutionResult,
    classify_execution_outcome,
    execute_sample_tests,
)


def test_heuristic_reclassification_promotes_shared_worker_error() -> None:
    shared = "worker exited with rc=137: OOM"
    results = (
        TestCaseResultProjection(
            input_value=(1,),
            expected_output=True,
            passed=False,
            error=shared,
        ),
        TestCaseResultProjection(
            input_value=(2,),
            expected_output=False,
            passed=False,
            error=shared,
        ),
    )
    classified = classify_execution_outcome(results)
    assert classified.outcome_kind == "infra_error"
    assert classified.infra_error is not None
    assert classified.infra_error.stage == "heuristic_reclassification"
    assert classified.infra_error.detail == shared


def test_distinct_compile_errors_stay_tested() -> None:
    results = (
        TestCaseResultProjection(
            input_value=(1,),
            expected_output=True,
            passed=False,
            compile_error="SyntaxError: line 1",
        ),
        TestCaseResultProjection(
            input_value=(2,),
            expected_output=False,
            passed=False,
            compile_error="SyntaxError: line 2",
        ),
    )
    classified = classify_execution_outcome(results)
    assert classified.outcome_kind == "tested"


def test_successful_execution_returns_tested_with_results() -> None:
    result = execute_sample_tests(
        extracted_code="def add(a, b):\n    return a + b\n",
        entry_point="add",
        test_cases=[TestCase(input_value=(1, 2), expected_output=3)],
        timeout_seconds=1.0,
        sample_id="sample-3",
    )

    assert result.outcome_kind == "tested"
    assert result.test_pass_rate == 1.0
    assert len(result.test_case_results) == 1
    assert result.infra_error is None


def test_syntax_error_returns_per_case_failures() -> None:
    result = execute_sample_tests(
        extracted_code="def nope(:\n    return 1\n",
        entry_point="nope",
        test_cases=[TestCase(input_value=(), expected_output=1)],
        timeout_seconds=1.0,
        sample_id="syntax-error",
    )

    assert result.outcome_kind == "tested"
    assert result.test_pass_rate == 0.0
    assert result.test_case_results[0].compile_success is False
    assert result.test_case_results[0].compile_error is not None


def test_execution_resets_namespace_between_samples() -> None:
    code = (
        "counter = 0\n"
        "def value():\n"
        "    global counter\n"
        "    counter += 1\n"
        "    return counter\n"
    )
    first = execute_sample_tests(
        extracted_code=code,
        entry_point="value",
        test_cases=[
            TestCase(input_value=(), expected_output=1),
            TestCase(input_value=(), expected_output=2),
        ],
        timeout_seconds=1.0,
        sample_id="namespace-reset-1",
    )
    second = execute_sample_tests(
        extracted_code=code,
        entry_point="value",
        test_cases=[TestCase(input_value=(), expected_output=1)],
        timeout_seconds=1.0,
        sample_id="namespace-reset-2",
    )

    assert first.outcome_kind == "tested"
    assert first.test_pass_rate == 1.0
    assert [case.actual_output for case in first.test_case_results] == [1, 2]
    assert second.outcome_kind == "tested"
    assert second.test_pass_rate == 1.0
    assert [case.actual_output for case in second.test_case_results] == [1]


def test_printed_output_does_not_corrupt_worker_payload() -> None:
    result = execute_sample_tests(
        extracted_code=(
            "def noisy(x):\n"
            "    print('hello from candidate')\n"
            "    return x\n"
        ),
        entry_point="noisy",
        test_cases=[TestCase(input_value=(3,), expected_output=3)],
        timeout_seconds=1.0,
        sample_id="noisy",
    )

    assert result.outcome_kind == "tested"
    assert result.test_pass_rate == 1.0


def test_timeout_maps_to_structured_infra_result() -> None:
    result = execute_sample_tests(
        extracted_code=(
            "def spin():\n"
            "    while True:\n"
            "        pass\n"
        ),
        entry_point="spin",
        test_cases=[TestCase(input_value=(), expected_output=1)],
        timeout_seconds=0.1,
        sample_id="timeout",
    )

    assert isinstance(result, SampleExecutionResult)
    assert result.outcome_kind == "infra_error"
    assert result.test_case_results == ()
    assert result.test_pass_rate is None
    assert result.infra_error == InfraErrorProjection(
        stage="worker_timeout",
        execution_mode="local_fork_worker",
        detail="worker timed out after 0.1s",
    )
