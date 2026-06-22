"""Contained nl-code Docker execution for one sample."""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass
from typing import Literal

from nl_code.code_execution.models import (
    CodeExecutionInfrastructureError,
    TestCase,
    TestCaseResult,
)
from nl_code.code_execution.runner import run_test_cases

from dr_code.models.outcomes import InfraErrorProjection, TestCaseResultProjection
from dr_code.testing.bridge import build_eval_code, run_function_name

logger = logging.getLogger("dr_code.testing")

ExecutionOutcomeKind = Literal["tested", "infra_error", "internal_error"]

_INFRA_MESSAGE_MARKERS = (
    "execution infrastructure failure",
    "worker exited",
    "worker returned no output",
    "docker_unavailable",
    "docker_timeout",
    "invalid JSON from worker",
    "missing results list",
)


@dataclass(frozen=True)
class SampleExecutionResult:
    """Result of executing one sample's test cases."""

    outcome_kind: ExecutionOutcomeKind
    test_case_results: tuple[TestCaseResultProjection, ...]
    test_pass_rate: float | None
    infra_error: InfraErrorProjection | None
    internal_error: str | None
    latency_ms: float


def execute_sample_tests(
    *,
    extracted_code: str,
    entry_point: str,
    test_cases: list[TestCase],
    timeout_seconds: float,
    docker_image: str | None,
    sample_id: str,
) -> SampleExecutionResult:
    """Run one sample in an isolated Docker container with full containment."""
    started = time.perf_counter()
    logger.info("%s execute_sample_tests start", sample_id)
    eval_code = build_eval_code(extracted_code, entry_point)

    try:
        raw_results, pass_rate = run_test_cases(
            code=eval_code,
            function_name=run_function_name(),
            test_cases=test_cases,
            timeout_seconds=timeout_seconds,
            docker_image=docker_image,
        )
    except CodeExecutionInfrastructureError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        result = _infra_result(
            infra_error=_project_infra_error(exc),
            latency_ms=latency_ms,
        )
        logger.info(
            "%s execute_sample_tests end outcome_kind=%s latency_ms=%.2f",
            sample_id,
            result.outcome_kind,
            result.latency_ms,
        )
        return result
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        logger.error(
            "%s execute_sample_tests internal_error type=%s",
            sample_id,
            type(exc).__name__,
            exc_info=True,
        )
        result = SampleExecutionResult(
            outcome_kind="internal_error",
            test_case_results=(),
            test_pass_rate=None,
            infra_error=None,
            internal_error=(
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            ),
            latency_ms=latency_ms,
        )
        logger.info(
            "%s execute_sample_tests end outcome_kind=%s latency_ms=%.2f",
            sample_id,
            result.outcome_kind,
            result.latency_ms,
        )
        return result

    projected = tuple(_project_test_case(result) for result in raw_results)
    classified = classify_execution_outcome(projected)
    if classified.outcome_kind == "infra_error":
        logger.warning(
            "%s heuristic infra reclassification detail=%s",
            sample_id,
            classified.infra_error.detail if classified.infra_error else "",
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    if classified.outcome_kind == "tested":
        result = SampleExecutionResult(
            outcome_kind="tested",
            test_case_results=projected,
            test_pass_rate=pass_rate,
            infra_error=None,
            internal_error=None,
            latency_ms=latency_ms,
        )
    else:
        result = SampleExecutionResult(
            outcome_kind="infra_error",
            test_case_results=(),
            test_pass_rate=None,
            infra_error=classified.infra_error,
            internal_error=None,
            latency_ms=latency_ms,
        )

    logger.info(
        "%s execute_sample_tests end outcome_kind=%s latency_ms=%.2f",
        sample_id,
        result.outcome_kind,
        result.latency_ms,
    )
    return result


@dataclass(frozen=True)
class _Classification:
    outcome_kind: ExecutionOutcomeKind
    infra_error: InfraErrorProjection | None = None


def classify_execution_outcome(
    results: tuple[TestCaseResultProjection, ...],
) -> _Classification:
    """Reclassify worker-level failures misreported as per-case test failures."""
    if not results:
        return _Classification(outcome_kind="tested")

    if any(result.passed for result in results):
        return _Classification(outcome_kind="tested")

    messages: list[str] = []
    for result in results:
        message = result.error or result.compile_error
        if message is None:
            return _Classification(outcome_kind="tested")
        messages.append(message)

    if len(set(messages)) != 1:
        return _Classification(outcome_kind="tested")

    shared = messages[0]
    if not _looks_like_infra_message(shared):
        return _Classification(outcome_kind="tested")

    return _Classification(
        outcome_kind="infra_error",
        infra_error=InfraErrorProjection(
            stage="heuristic_reclassification",
            execution_mode="docker_worker",
            detail=shared,
        ),
    )


def _looks_like_infra_message(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _INFRA_MESSAGE_MARKERS)


def _project_infra_error(
    exc: CodeExecutionInfrastructureError,
) -> InfraErrorProjection:
    return InfraErrorProjection(
        stage=exc.stage,
        execution_mode=exc.execution_mode,
        detail=exc.detail,
    )


def _project_test_case(result: TestCaseResult) -> TestCaseResultProjection:
    return TestCaseResultProjection(
        input_value=result.input_value,
        expected_output=result.expected_output,
        actual_output=result.actual_output,
        passed=result.passed,
        error=result.error,
        compile_success=result.compile_success,
        compile_error=result.compile_error,
    )


def _infra_result(
    *,
    infra_error: InfraErrorProjection,
    latency_ms: float,
) -> SampleExecutionResult:
    return SampleExecutionResult(
        outcome_kind="infra_error",
        test_case_results=(),
        test_pass_rate=None,
        infra_error=infra_error,
        internal_error=None,
        latency_ms=latency_ms,
    )
