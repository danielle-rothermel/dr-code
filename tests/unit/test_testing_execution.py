"""Unit tests for execute_sample_tests containment and classification."""

from __future__ import annotations

from unittest.mock import patch

from nl_code.code_execution.models import (
    CodeExecutionInfrastructureError,
    TestCase,
    TestCaseResult,
)

from dr_code.models.outcomes import InfraErrorProjection, TestCaseResultProjection
from dr_code.testing.execution import (
    SampleExecutionResult,
    classify_execution_outcome,
    execute_sample_tests,
)


def test_infra_exception_maps_to_structured_infra_result() -> None:
    exc = CodeExecutionInfrastructureError(
        stage="docker_timeout",
        execution_mode="docker_worker",
        detail="timed out",
    )

    with patch(
        "dr_code.testing.execution.run_test_cases",
        side_effect=exc,
    ):
        result = execute_sample_tests(
            extracted_code="def f():\n    return 1\n",
            entry_point="f",
            test_cases=[TestCase(input_value=(), expected_output=1)],
            timeout_seconds=1.0,
            docker_image=None,
            sample_id="sample-1",
        )

    assert result.outcome_kind == "infra_error"
    assert result.test_case_results == ()
    assert result.test_pass_rate is None
    assert result.infra_error == InfraErrorProjection(
        stage="docker_timeout",
        execution_mode="docker_worker",
        detail="timed out",
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


def test_unexpected_exception_returns_internal_error() -> None:
    with patch(
        "dr_code.testing.execution.run_test_cases",
        side_effect=RuntimeError("boom"),
    ):
        result = execute_sample_tests(
            extracted_code="def f():\n    return 1\n",
            entry_point="f",
            test_cases=[TestCase(input_value=(), expected_output=1)],
            timeout_seconds=1.0,
            docker_image=None,
            sample_id="sample-2",
        )

    assert result.outcome_kind == "internal_error"
    assert result.test_case_results == ()
    assert result.test_pass_rate is None
    assert result.internal_error is not None
    assert "RuntimeError: boom" in result.internal_error


def test_successful_execution_returns_tested_with_results() -> None:
    fake_results = [
        TestCaseResult(
            input_value=(1, 2),
            expected_output=3,
            actual_output=3,
            passed=True,
        ),
    ]

    with patch(
        "dr_code.testing.execution.run_test_cases",
        return_value=(fake_results, 1.0),
    ):
        result = execute_sample_tests(
            extracted_code="def add(a, b):\n    return a + b\n",
            entry_point="add",
            test_cases=[TestCase(input_value=(1, 2), expected_output=3)],
            timeout_seconds=1.0,
            docker_image=None,
            sample_id="sample-3",
        )

    assert result.outcome_kind == "tested"
    assert result.test_pass_rate == 1.0
    assert len(result.test_case_results) == 1
    assert result.infra_error is None


def test_infra_result_never_sets_pass_rate() -> None:
    exc = CodeExecutionInfrastructureError(
        stage="docker_unavailable",
        execution_mode="docker_worker",
        detail="daemon down",
    )
    with patch(
        "dr_code.testing.execution.run_test_cases",
        side_effect=exc,
    ):
        result = execute_sample_tests(
            extracted_code="pass",
            entry_point="f",
            test_cases=[TestCase(input_value=(), expected_output=1)],
            timeout_seconds=1.0,
            docker_image=None,
            sample_id="sample-4",
        )
    assert isinstance(result, SampleExecutionResult)
    assert result.test_pass_rate is None
