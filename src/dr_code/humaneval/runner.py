from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Final

from pydantic import TypeAdapter, ValidationError

from dr_exec import Executor

from dr_code.humaneval.parsed_tests import (
    ParsedTests,
    SingleCaseCheck,
    TestCase,
)
from dr_code.core.execution.executor import (
    CompletedPythonProcess,
    ExecutionKilledError,
    ExecutionOutputLimitError,
    ExecutionTimeoutError,
    run_python_source,
)
from dr_code.humaneval.task import (
    EvaluationCaseResult,
    EvaluationCaseStatus,
    EvaluationHarnessError,
    EvaluationTaskResult,
    HumanEvalRunnerCandidateFailure,
    HumanEvalRunnerCaseResults,
    HumanEvalRunnerHarnessFailure,
    HumanEvalRunnerOutput,
    HumanEvalRunnerPayload,
    HumanEvalTask,
)

HUMANEVAL_RUNNER_COMPUTATION_ID: Final = "humaneval-runner@0"


@dataclass(frozen=True, slots=True)
class HumanEvalBatchRequest:
    source: str
    input_json: str
    timeout_seconds: float


def evaluate_humaneval_code(
    *,
    task: HumanEvalTask,
    candidate_code: str,
    timeout_seconds: float,
    candidate_ast: ast.Module | None = None,
    executor: Executor | None = None,
) -> EvaluationTaskResult:
    parsed_tests = require_parsed_tests(task)
    function_names = top_level_function_names(
        candidate_code,
        parsed_module=candidate_ast,
    )
    checks = list(parsed_tests.iter_checks(candidate_name="candidate"))
    runner_source = runner_script()
    results: list[EvaluationCaseResult] = []
    for function_name in function_names:
        try:
            results.extend(
                run_subprocess_batch(
                    task=task,
                    candidate_code=candidate_code,
                    function_name=function_name,
                    timeout_seconds=timeout_seconds,
                    checks=checks,
                    runner_source=runner_source,
                    executor=executor,
                )
            )
        except EvaluationHarnessError as exc:
            evaluation = EvaluationTaskResult(
                task_id=task.task_id,
                entry_point=task.entry_point,
                function_names=function_names,
                total_cases=len(parsed_tests.cases),
                results=[*results, *exc.case_results],
            )
            raise EvaluationHarnessError(
                str(exc),
                case_results=exc.case_results,
                evaluation=evaluation,
                cause=exc.cause or exc,
            ) from exc
    return EvaluationTaskResult(
        task_id=task.task_id,
        entry_point=task.entry_point,
        function_names=function_names,
        total_cases=len(parsed_tests.cases),
        results=results,
    )


def top_level_function_names(
    code_str: str,
    *,
    parsed_module: ast.Module | None = None,
) -> list[str]:
    tree = parsed_module if parsed_module is not None else ast.parse(code_str)
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]


def run_subprocess_batch(
    *,
    task: HumanEvalTask,
    candidate_code: str,
    function_name: str,
    timeout_seconds: float,
    checks: list[SingleCaseCheck] | None = None,
    runner_source: str | None = None,
    executor: Executor | None = None,
) -> list[EvaluationCaseResult]:
    request = build_humaneval_batch_request(
        task=task,
        candidate_code=candidate_code,
        function_name=function_name,
        timeout_seconds=timeout_seconds,
        checks=checks,
        runner_source=runner_source,
    )
    started_at = time.perf_counter()
    try:
        completed = run_python_source(
            executor,
            source=request.source,
            input_json=request.input_json,
            timeout_seconds=request.timeout_seconds,
        )
    except ExecutionTimeoutError:
        return timeout_results(
            task=task,
            function_name=function_name,
            timeout_seconds=request.timeout_seconds,
        )
    except (ExecutionKilledError, ExecutionOutputLimitError) as exc:
        return error_results(
            task=task,
            function_name=function_name,
            message=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.perf_counter() - started_at,
        )
    except Exception as exc:
        elapsed_seconds = time.perf_counter() - started_at
        case_results = error_results(
            task=task,
            function_name=function_name,
            message=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=elapsed_seconds,
        )
        raise EvaluationHarnessError(
            "subprocess execution failed",
            case_results=case_results,
            cause=exc,
        ) from exc

    return interpret_subprocess_batch_result(
        task=task,
        function_name=function_name,
        completed=completed,
        elapsed_seconds=time.perf_counter() - started_at,
    )


def build_humaneval_batch_request(
    *,
    task: HumanEvalTask,
    candidate_code: str,
    function_name: str,
    timeout_seconds: float,
    checks: list[SingleCaseCheck] | None = None,
    runner_source: str | None = None,
) -> HumanEvalBatchRequest:
    parsed_tests = require_parsed_tests(task)
    check_payloads = (
        checks
        if checks is not None
        else list(parsed_tests.iter_checks(candidate_name="candidate"))
    )
    payload = HumanEvalRunnerPayload(
        task_id=task.task_id,
        candidate_code=candidate_code,
        support_code=parsed_tests.support_code,
        function_name=function_name,
        test_type=parsed_tests.test_type,
        checks=check_payloads,
    )
    return HumanEvalBatchRequest(
        source=runner_source or runner_script(),
        input_json=payload.model_dump_json(),
        timeout_seconds=timeout_seconds,
    )


def interpret_subprocess_batch_result(
    *,
    task: HumanEvalTask,
    function_name: str,
    completed: CompletedPythonProcess,
    elapsed_seconds: float,
) -> list[EvaluationCaseResult]:
    """Allow partial output, but require known and unique returned case IDs.

    Candidate-attributable kills never reach this interpretation: they
    surface as typed `ExecutionKilledError` classifications before a
    completed process exists. A nonzero exit that completes the protected
    protocol is runner breakage, not candidate evidence.
    """

    parsed_tests = require_parsed_tests(task)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        case_results = error_results(
            task=task,
            function_name=function_name,
            message=message,
            elapsed_seconds=elapsed_seconds,
        )
        raise EvaluationHarnessError(
            "runner subprocess exited nonzero"
            + (f": {message}" if message else ""),
            case_results=case_results,
        )
    try:
        raw_results = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        case_results = error_results(
            task=task,
            function_name=function_name,
            message=f"Could not decode runner output: {exc}",
            elapsed_seconds=elapsed_seconds,
        )
        raise EvaluationHarnessError(
            f"runner output was not valid JSON: {exc}",
            case_results=case_results,
            cause=exc,
        ) from exc
    adapter = TypeAdapter(HumanEvalRunnerOutput)
    try:
        runner_output = adapter.validate_python(raw_results)
    except ValidationError as exc:
        case_results = error_results(
            task=task,
            function_name=function_name,
            message=f"Invalid runner output: {exc}",
            elapsed_seconds=elapsed_seconds,
        )
        raise EvaluationHarnessError(
            f"runner output failed validation: {exc}",
            case_results=case_results,
            cause=exc,
        ) from exc

    if isinstance(runner_output, HumanEvalRunnerHarnessFailure):
        case_results = error_results(
            task=task,
            function_name=function_name,
            message=runner_output.message,
            elapsed_seconds=runner_output.elapsed_seconds,
        )
        raise EvaluationHarnessError(
            "runner support-code initialization failed: "
            + runner_output.message,
            case_results=case_results,
        )
    if isinstance(runner_output, HumanEvalRunnerCandidateFailure):
        return error_results(
            task=task,
            function_name=function_name,
            message=runner_output.message,
            elapsed_seconds=runner_output.elapsed_seconds,
        )
    assert isinstance(runner_output, HumanEvalRunnerCaseResults)

    expected_case_ids = {case.case_id for case in parsed_tests.cases}
    seen_case_ids: set[str] = set()
    results: list[EvaluationCaseResult] = []
    for runner_result in runner_output.results:
        if (
            runner_result.case_id not in expected_case_ids
            or runner_result.case_id in seen_case_ids
        ):
            results.append(
                EvaluationCaseResult(
                    task_id=task.task_id,
                    case_id=runner_result.case_id,
                    function_name=function_name,
                    status=EvaluationCaseStatus.ERROR,
                    message=(
                        "Invalid runner output: duplicate or unknown case id "
                        f"{runner_result.case_id!r}"
                    ),
                    test_type=parsed_tests.test_type,
                    elapsed_seconds=elapsed_seconds,
                )
            )
            raise EvaluationHarnessError(
                "runner output contained duplicate or unknown case ids",
                case_results=results,
            )
        seen_case_ids.add(runner_result.case_id)
        results.append(
            EvaluationCaseResult(
                task_id=task.task_id,
                case_id=runner_result.case_id,
                function_name=function_name,
                status=runner_result.status,
                message=runner_result.message,
                test_type=parsed_tests.test_type,
                input_repr=runner_result.input_repr,
                expected_output_repr=runner_result.expected_output_repr,
                actual_output_repr=runner_result.actual_output_repr,
                elapsed_seconds=runner_result.elapsed_seconds,
                timeout_seconds=runner_result.timeout_seconds,
            )
        )
    return results


def timeout_results(
    *,
    task: HumanEvalTask,
    function_name: str,
    timeout_seconds: float,
) -> list[EvaluationCaseResult]:
    parsed_tests = require_parsed_tests(task)
    results: list[EvaluationCaseResult] = []
    for case in parsed_tests.cases:
        metadata = case_metadata(parsed_tests, case)
        results.append(
            EvaluationCaseResult(
                task_id=task.task_id,
                case_id=case.case_id,
                function_name=function_name,
                status=EvaluationCaseStatus.TIMEOUT,
                message=f"Batch timed out after {timeout_seconds} seconds",
                test_type=parsed_tests.test_type,
                input_repr=metadata["input_repr"],
                expected_output_repr=metadata["expected_output_repr"],
                actual_output_repr=metadata["actual_output_repr"],
                elapsed_seconds=timeout_seconds,
                timeout_seconds=timeout_seconds,
            )
        )
    return results


def error_results(
    *,
    task: HumanEvalTask,
    function_name: str,
    message: str,
    elapsed_seconds: float | None = None,
) -> list[EvaluationCaseResult]:
    parsed_tests = require_parsed_tests(task)
    results: list[EvaluationCaseResult] = []
    for case in parsed_tests.cases:
        metadata = case_metadata(parsed_tests, case)
        results.append(
            EvaluationCaseResult(
                task_id=task.task_id,
                case_id=case.case_id,
                function_name=function_name,
                status=EvaluationCaseStatus.ERROR,
                message=message,
                test_type=parsed_tests.test_type,
                input_repr=metadata["input_repr"],
                expected_output_repr=metadata["expected_output_repr"],
                actual_output_repr=metadata["actual_output_repr"],
                elapsed_seconds=elapsed_seconds,
            )
        )
    return results


def case_metadata(
    parsed_tests: ParsedTests,
    case: TestCase,
) -> dict[str, str]:
    check = case.as_check(
        candidate_name="candidate",
        assertion_name=parsed_tests.assertion_name,
    )
    return {
        "input_repr": check.input_repr,
        "expected_output_repr": check.expected_output_repr,
        "actual_output_repr": "",
    }


def require_parsed_tests(task: HumanEvalTask) -> ParsedTests:
    if task.parsed_tests is None:
        raise ValueError("HumanEvalTask.parsed_tests is required")
    return task.parsed_tests


@cache
def runner_script() -> str:
    # Redirected stdout prevents accidental collisions, not result forgery.
    return (
        files("dr_code.humaneval")
        .joinpath("runner_driver_script.py")
        .read_text(encoding="utf-8")
    )
