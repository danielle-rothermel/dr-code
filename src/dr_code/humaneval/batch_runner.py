"""Subprocess batch orchestration for HumanEval evaluation.

Owns the single canonical HumanEval batch protocol: ``build_humaneval_batch_
request`` builds every request, and ``interpret_subprocess_batch_result``
interprets every completed process back into ``EvaluationCaseResult`` rows.
Both the direct batch path here and the ``code_test`` metrics operator route
through them, so the request bytes and the result reading have one
implementation.

The runner validates each returned case result, but it currently preserves
partial runner output rather than requiring one returned row per parsed test
case. Tightening that cardinality check would be a benchmark behavior change
and is deferred until per-test score persistence semantics are defined.
Returned case ids must still be known and unique so partial output can never
inflate coverage.

Failure attribution: candidate-attributable terminations (memory/CPU-limit
SIGKILL, interpreter crash, SystemExit, output floods) are scored as case
errors or timeouts; ``EvaluationHarnessError``/``HarnessFailure`` is reserved
for sandbox or runtime breakage so operators can alert on it. The runner
captures the protocol stdout handle before candidate code runs and redirects
Python-level candidate output to stderr, so candidate prints cannot reach the
results channel.
"""

from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass
from functools import cache
from importlib.resources import files

from pydantic import TypeAdapter, ValidationError

from dr_code.humaneval.parsed_tests import (
    ParsedTests,
    SingleCaseCheck,
    TestCase,
)
from dr_code.humaneval.sandbox import (
    CANDIDATE_KILL_RETURNCODES,
    SandboxCompletedProcess,
    SandboxOutputLimitError,
    SandboxRunner,
    SandboxTimeoutError,
    run_python_in_sandbox,
)
from dr_code.humaneval.task import (
    EvaluationCaseResult,
    EvaluationCaseStatus,
    EvaluationHarnessError,
    EvaluationTaskResult,
    HumanEvalRunnerCaseOutput,
    HumanEvalRunnerPayload,
    HumanEvalTask,
)


@dataclass(frozen=True, slots=True)
class HumanEvalBatchRequest:
    """Trusted runner source and opaque input for one HumanEval batch."""

    source: str
    input_json: str
    timeout_seconds: float


def evaluate_humaneval_code(
    *,
    task: HumanEvalTask,
    candidate_code: str,
    timeout_seconds: float,
    candidate_ast: ast.Module | None = None,
    run_in_sandbox: SandboxRunner = run_python_in_sandbox,
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
                    run_in_sandbox=run_in_sandbox,
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
    """Every top-level function defined by candidate source, in order.

    Legal duplicate names are all returned, so their case results stack under
    one name downstream and can prevent ``coverage_complete``.
    """

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
    run_in_sandbox: SandboxRunner = run_python_in_sandbox,
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
        completed = run_in_sandbox(
            source=request.source,
            input_json=request.input_json,
            timeout_seconds=request.timeout_seconds,
        )
    except SandboxTimeoutError:
        return timeout_results(
            task=task,
            function_name=function_name,
            timeout_seconds=request.timeout_seconds,
        )
    except SandboxOutputLimitError as exc:
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
    """Build the complete sandbox request for one HumanEval batch.

    The one place a HumanEval runner payload is assembled. Every caller --
    the direct batch path and the ``code_test`` operator -- gets byte-identical
    request input for the same task, candidate, and function name.
    """

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
    completed: SandboxCompletedProcess,
    elapsed_seconds: float,
) -> list[EvaluationCaseResult]:
    """Read one completed process through the HumanEval runner protocol.

    The one place runner output is turned into case results. Returncode
    attribution, JSON decoding, per-case validation, and the known/unique
    case-id rule all live here, so the direct batch path and the ``code_test``
    operator read a given process identically.
    """

    parsed_tests = require_parsed_tests(task)
    if completed.returncode in CANDIDATE_KILL_RETURNCODES:
        message = (
            f"sandbox killed candidate execution (exit {completed.returncode}"
            ": memory limit, CPU limit, or interpreter crash)"
        )
        detail = completed.stderr.strip()
        if detail:
            message = f"{message}: {detail}"
        return error_results(
            task=task,
            function_name=function_name,
            message=message,
            elapsed_seconds=elapsed_seconds,
        )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        case_results = error_results(
            task=task,
            function_name=function_name,
            message=message,
            elapsed_seconds=elapsed_seconds,
        )
        raise EvaluationHarnessError(
            "runner subprocess exited nonzero",
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
            "runner output was not valid JSON",
            case_results=case_results,
            cause=exc,
        ) from exc
    if not isinstance(raw_results, list):
        case_results = error_results(
            task=task,
            function_name=function_name,
            message=(
                "Invalid runner output: expected a JSON list of case results"
            ),
            elapsed_seconds=elapsed_seconds,
        )
        raise EvaluationHarnessError(
            "runner output had invalid shape",
            case_results=case_results,
        )

    adapter = TypeAdapter(HumanEvalRunnerCaseOutput)
    expected_case_ids = {case.case_id for case in parsed_tests.cases}
    seen_case_ids: set[str] = set()
    results: list[EvaluationCaseResult] = []
    for item in raw_results:
        try:
            runner_result = adapter.validate_python(item)
        except ValidationError as exc:
            case_id = (
                str(item["case_id"])
                if isinstance(item, dict) and "case_id" in item
                else f"case_{len(results)}"
            )
            metadata: dict[str, str] = {}
            for case in parsed_tests.cases:
                if case.case_id == case_id:
                    metadata = case_metadata(parsed_tests, case)
                    break
            results.append(
                EvaluationCaseResult(
                    task_id=task.task_id,
                    case_id=case_id,
                    function_name=function_name,
                    status=EvaluationCaseStatus.ERROR,
                    message=f"Invalid runner output: {exc}",
                    test_type=parsed_tests.test_type,
                    input_repr=metadata.get("input_repr", ""),
                    expected_output_repr=metadata.get(
                        "expected_output_repr",
                        "",
                    ),
                    actual_output_repr=metadata.get(
                        "actual_output_repr",
                        "",
                    ),
                    elapsed_seconds=elapsed_seconds,
                )
            )
            raise EvaluationHarnessError(
                "runner output case failed validation",
                case_results=results,
                cause=exc,
            ) from exc
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
    # The standalone runner program lives in ``sandbox_runner_script.py`` and
    # is read as text (never imported) so it stays dependency-free and can run
    # interpreter-isolated inside the sandbox container. See that file's header.
    return (
        files("dr_code.humaneval")
        .joinpath("sandbox_runner_script.py")
        .read_text(encoding="utf-8")
    )
