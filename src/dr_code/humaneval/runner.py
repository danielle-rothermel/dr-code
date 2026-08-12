from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from dr_code.humaneval.parsed_tests import ParsedTests, TestCase
from dr_code.humaneval.task import (
    EvaluationCaseResult,
    EvaluationCaseStatus,
    EvaluationHarnessError,
    EvaluationTaskResult,
    HumanEvalTask,
)

if TYPE_CHECKING:
    from dr_code.evaluation.records import CandidateExecutionOutcome
    from dr_code.humaneval.job import HumanEvalCandidateJobResult


def evaluation_from_candidate_execution(
    *,
    task: HumanEvalTask,
    candidate_source: str,
    question: object,
    outcome: CandidateExecutionOutcome,
) -> EvaluationTaskResult:
    from dr_code.evaluation.records import (
        CandidateJobCompleted,
        CandidateJobTerminated,
        CandidateTerminationReason,
        ExecutorExecutionFailure,
        HarnessExecutionFailure,
    )
    from dr_code.metrics import MetricQuestionCoordinate

    if not isinstance(question, MetricQuestionCoordinate):
        raise TypeError("question must be a MetricQuestionCoordinate")
    fallback_names = _safe_top_level_function_names(candidate_source)
    if isinstance(outcome, CandidateJobTerminated):
        results: list[EvaluationCaseResult] = []
        for function_name in fallback_names:
            if outcome.reason is CandidateTerminationReason.WALL_TIME:
                results.extend(
                    timeout_results(
                        task=task,
                        function_name=function_name,
                    )
                )
            else:
                results.extend(
                    error_results(
                        task=task,
                        function_name=function_name,
                        message=(
                            "candidate execution terminated: "
                            f"{outcome.reason.value}"
                        ),
                    )
                )
        return EvaluationTaskResult(
            task_id=task.task_id,
            entry_point=task.entry_point,
            function_names=fallback_names,
            total_cases=len(task.parsed_tests.cases),
            results=results,
        )
    if isinstance(outcome, HarnessExecutionFailure | ExecutorExecutionFailure):
        raise EvaluationHarnessError(
            f"{outcome.failure_type}: {outcome.message}"
        )
    if not isinstance(outcome, CandidateJobCompleted):
        raise EvaluationHarnessError(
            f"unsupported candidate execution outcome: {type(outcome).__name__}"
        )
    return _evaluation_from_job_result(
        task=task,
        fallback_names=fallback_names,
        question=question,
        result=outcome.result,
    )


def _evaluation_from_job_result(
    *,
    task: HumanEvalTask,
    fallback_names: list[str],
    question: object,
    result: HumanEvalCandidateJobResult,
) -> EvaluationTaskResult:
    from dr_code.humaneval.job import (
        CandidateNamespaceFailure,
        HumanEvalSuiteCompleted,
        HumanEvalSuiteHarnessFailure,
    )

    if isinstance(result.namespace, CandidateNamespaceFailure):
        case_results = [
            case
            for function_name in fallback_names
            for case in error_results(
                task=task,
                function_name=function_name,
                message=(
                    f"{result.namespace.failure_type}: "
                    f"{result.namespace.message}"
                ),
            )
        ]
        return EvaluationTaskResult(
            task_id=task.task_id,
            entry_point=task.entry_point,
            function_names=fallback_names,
            total_cases=len(task.parsed_tests.cases),
            results=case_results,
        )
    suite = next(
        (suite for suite in result.suites if suite.question == question),
        None,
    )
    if suite is None:
        raise EvaluationHarnessError(
            "candidate job returned no matching HumanEval suite"
        )
    groups = (
        suite.groups
        if isinstance(suite, HumanEvalSuiteCompleted)
        else suite.completed_groups
    )
    case_results = [case for group in groups for case in group.cases]
    evaluation = EvaluationTaskResult(
        task_id=task.task_id,
        entry_point=task.entry_point,
        function_names=list(result.namespace.function_names),
        total_cases=len(task.parsed_tests.cases),
        results=case_results,
    )
    if isinstance(suite, HumanEvalSuiteHarnessFailure):
        raise EvaluationHarnessError(
            f"{suite.failure_type}: {suite.message}",
            case_results=case_results,
            evaluation=evaluation,
        )
    return evaluation


def top_level_function_names(
    code_str: str,
    *,
    parsed_module: ast.Module | None = None,
) -> list[str]:
    tree = parsed_module if parsed_module is not None else ast.parse(code_str)
    function_names: list[str] = []
    seen_names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name in seen_names:
            continue
        seen_names.add(node.name)
        function_names.append(node.name)
    return function_names


def _safe_top_level_function_names(source: str) -> list[str]:
    try:
        return top_level_function_names(source)
    except SyntaxError:
        return []


def timeout_results(
    *,
    task: HumanEvalTask,
    function_name: str,
) -> list[EvaluationCaseResult]:
    parsed_tests = task.parsed_tests
    results: list[EvaluationCaseResult] = []
    for case in parsed_tests.cases:
        metadata = case_metadata(parsed_tests, case)
        results.append(
            EvaluationCaseResult(
                task_id=task.task_id,
                case_id=case.case_id,
                function_name=function_name,
                status=EvaluationCaseStatus.TIMEOUT,
                message="Candidate job exceeded its wall-time budget",
                test_type=parsed_tests.test_type,
                input_repr=metadata["input_repr"],
                expected_output_repr=metadata["expected_output_repr"],
                actual_output_repr=metadata["actual_output_repr"],
                elapsed_seconds=None,
                timeout_seconds=None,
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
    parsed_tests = task.parsed_tests
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


__all__ = [
    "case_metadata",
    "error_results",
    "evaluation_from_candidate_execution",
    "timeout_results",
    "top_level_function_names",
]
