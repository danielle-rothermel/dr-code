"""HumanEval execution over dr-exec's batch driver kit and NDJSON protocol.

The adapter owns HumanEval case semantics and candidate-versus-harness failure
attribution; dr-exec owns spawn, budgets, capture, batch transport, and the
per-item NDJSON delivery that makes partial results survive a child's death.

One batch is one function's cases run in one warm child: each case is a
``BatchItem`` whose result the kit emits the moment it is produced, so a late
death, a wall-clock deadline, or an output overflow costs only the unfinished
tail — completed cases are already delivered. The parent branches on dr-exec's
attribution *before* reading any transcript: a budget or infrastructure outcome
is decided as data, never mistaken for a protocol error.

Candidate code has the worker's filesystem, credential, process, and network
permissions; direct file-descriptor writes to the protocol channel are a
declared limit of the containment profile. Run it only on disposable workers
constrained outside this process boundary.
"""

from __future__ import annotations

import ast
import json
import signal
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Final, Protocol

from dr_exec import (
    HERMETIC,
    PROCESS_BOUNDARY_ONLY,
    Attribution,
    BatchItem,
    BatchRequest,
    BatchResult,
    BudgetAxis,
    Budgets,
    ContainmentProfile,
    EnvironmentGrant,
    ItemResult,
    OutputBudget,
    OverflowPolicy,
    ProtocolChannelBudget,
    PythonRuntime,
    Records,
    RunResult,
    run_batch,
)
from pydantic import TypeAdapter, ValidationError

from dr_code.humaneval.parsed_tests import (
    ParsedTests,
    SingleCaseCheck,
    TestCase,
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

# dr-code's declared execution budgets, at the site where the protocol
# knowledge lives (adjudication artifact 4). Output overflow is FAIL: the
# payload stream is killed and budget-attributed, and dr-exec retains the
# captured-so-far bytes with a truncation mark.
MAX_HUMANEVAL_INPUT_BYTES: Final[int] = 4 * 1024 * 1024
MAX_HUMANEVAL_OUTPUT_BYTES: Final[int] = 1024 * 1024

# Per-case protocol result bound: one case's delivered JSON result line. The
# in-child clip keeps a flooding case's own result reportable rather than
# voiding the batch.
MAX_HUMANEVAL_ITEM_RESULT_BYTES: Final[int] = 128 * 1024

# The containment profile and runtime dr-code declares for HumanEval batches.
HUMANEVAL_PROFILE: Final[ContainmentProfile] = PROCESS_BOUNDARY_ONLY
HUMANEVAL_RUNTIME: Final[PythonRuntime] = HERMETIC

# A determinism and thread-oversubscription control, granted explicitly at the
# Python-execution call site (adjudication requirement 14).
HUMANEVAL_ENVIRONMENT: Final[EnvironmentGrant] = EnvironmentGrant.fixed(
    {"OPENBLAS_NUM_THREADS": "1"}
)

# The batch config schema the driver body may validate its payloads against.
_ITEM_SCHEMA: Final[str] = "humaneval-case@v1"

# Returncodes that mean the candidate's own process died, not a clean exit:
# an external SIGKILL or an interpreter SIGSEGV. Scored against the candidate.
CANDIDATE_KILL_RETURNCODES: Final[frozenset[int]] = frozenset(
    {-int(signal.SIGKILL), -int(signal.SIGSEGV)}
)


class BatchExecutor(Protocol):
    """The batch-running executor the adapter drives (real or fake)."""

    def run_batch(
        self,
        request: BatchRequest,
        *,
        profile: ContainmentProfile,
        budgets: Budgets,
        records: Records,
        runtime: PythonRuntime,
        environment: EnvironmentGrant,
    ) -> BatchResult: ...


class _ProductionExecutor:
    """Adapts dr-exec's ``run_batch`` entry point to the executor protocol.

    The real executor is a module function, not an object; this thin object
    lets the same call sites drive either it or a ``FakeExecutor`` without a
    branch. It claims no identity of its own — the run it produces carries
    ``EXECUTOR_IDENTITY``.
    """

    def run_batch(
        self,
        request: BatchRequest,
        *,
        profile: ContainmentProfile,
        budgets: Budgets,
        records: Records,
        runtime: PythonRuntime,
        environment: EnvironmentGrant,
    ) -> BatchResult:
        return run_batch(
            request,
            profile=profile,
            budgets=budgets,
            records=records,
            runtime=runtime,
            environment=environment,
        )


PRODUCTION_EXECUTOR: Final[_ProductionExecutor] = _ProductionExecutor()
"""The real dr-exec batch executor, wrapped for the executor protocol."""


@dataclass(frozen=True, slots=True)
class HumanEvalBatchPlan:
    """One function's dr-exec batch request plus the budgets it declares."""

    request: BatchRequest
    budgets: Budgets


def human_eval_budgets(
    timeout_seconds: float,
    *,
    output_bytes: int = MAX_HUMANEVAL_OUTPUT_BYTES,
    output_overflow_policy: OverflowPolicy = OverflowPolicy.FAIL,
    input_bytes: int = MAX_HUMANEVAL_INPUT_BYTES,
) -> Budgets:
    """dr-code's declared budgets for a HumanEval batch of one function."""
    return Budgets(
        wall_clock=timeout_seconds,
        output=OutputBudget(
            limit_bytes=output_bytes,
            overflow_policy=output_overflow_policy,
        ),
        input=input_bytes,
    )


def evaluate_human_eval_code(
    *,
    task: HumanEvalTask,
    candidate_code: str,
    timeout_seconds: float,
    executor: BatchExecutor,
    records: Records,
    candidate_ast: ast.Module | None = None,
) -> EvaluationTaskResult:
    parsed_tests = require_parsed_tests(task)
    function_names = top_level_function_names(
        candidate_code,
        parsed_module=candidate_ast,
    )
    checks = list(parsed_tests.iter_checks(candidate_name="candidate"))
    results: list[EvaluationCaseResult] = []
    for function_name in function_names:
        results.extend(
            run_function_batch(
                task=task,
                candidate_code=candidate_code,
                function_name=function_name,
                timeout_seconds=timeout_seconds,
                executor=executor,
                records=records,
                checks=checks,
            )
        )
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


def run_function_batch(
    *,
    task: HumanEvalTask,
    candidate_code: str,
    function_name: str,
    timeout_seconds: float,
    executor: BatchExecutor,
    records: Records,
    checks: list[SingleCaseCheck] | None = None,
) -> list[EvaluationCaseResult]:
    plan = build_human_eval_batch_plan(
        task=task,
        candidate_code=candidate_code,
        function_name=function_name,
        timeout_seconds=timeout_seconds,
        checks=checks,
    )
    result = executor.run_batch(
        plan.request,
        profile=HUMANEVAL_PROFILE,
        budgets=plan.budgets,
        records=records,
        runtime=HUMANEVAL_RUNTIME,
        environment=HUMANEVAL_ENVIRONMENT,
    )
    return interpret_batch_result(
        task=task,
        function_name=function_name,
        result=result,
        timeout_seconds=timeout_seconds,
    )


def build_human_eval_batch_plan(
    *,
    task: HumanEvalTask,
    candidate_code: str,
    function_name: str,
    timeout_seconds: float,
    budgets: Budgets | None = None,
    checks: list[SingleCaseCheck] | None = None,
) -> HumanEvalBatchPlan:
    """Build the dr-exec batch request and budgets for one function.

    ``budgets`` lets a caller declare its own output/input bounds (the
    code_test operator folds them into its identity); when omitted the lane's
    default HumanEval budgets are used.
    """
    parsed_tests = require_parsed_tests(task)
    check_payloads = (
        checks
        if checks is not None
        else list(parsed_tests.iter_checks(candidate_name="candidate"))
    )
    identity = HumanEvalRunnerPayload(
        task_id=task.task_id,
        candidate_code=candidate_code,
        support_code=parsed_tests.support_code,
        function_name=function_name,
        test_type=parsed_tests.test_type,
        checks=check_payloads,
    )
    items = tuple(
        BatchItem(item_id=check.case_id, payload=check.model_dump(mode="json"))
        for check in check_payloads
    )
    request = BatchRequest(
        items=items,
        body_source=compose_body(
            candidate_code=candidate_code,
            support_code=parsed_tests.support_code,
            function_name=function_name,
        ),
        item_schema=_ITEM_SCHEMA,
        config=json.loads(identity.model_dump_json()),
        channel_budget=ProtocolChannelBudget(
            item_result_bytes=MAX_HUMANEVAL_ITEM_RESULT_BYTES,
        ),
    )
    return HumanEvalBatchPlan(
        request=request,
        budgets=(
            budgets
            if budgets is not None
            else human_eval_budgets(timeout_seconds)
        ),
    )


def compose_body(
    *,
    candidate_code: str,
    support_code: str,
    function_name: str,
) -> str:
    """Bind the candidate/support/function literals into the driver body.

    The shared code travels baked into the body source rather than per item:
    every case in the batch runs the same candidate, so the body defines
    ``run_item`` over one namespace built once at load.
    """
    bindings = "\n".join(
        (
            f"_HE_CANDIDATE_CODE = {json.dumps(candidate_code)}",
            f"_HE_SUPPORT_CODE = {json.dumps(support_code)}",
            f"_HE_FUNCTION_NAME = {json.dumps(function_name)}",
        )
    )
    return f"{bindings}\n{driver_body_template()}"


def interpret_batch_result(
    *,
    task: HumanEvalTask,
    function_name: str,
    result: BatchResult,
    timeout_seconds: float,
) -> list[EvaluationCaseResult]:
    """Map one dr-exec batch result to HumanEval case results.

    The attribution pre-branch runs first: a wall-clock budget outcome is a
    per-case timeout, and an executor, channel, machine, or absence outcome is
    a harness failure that never scores against the candidate. Only then are
    the delivered per-item results read; a case that never reported is
    synthesized from the run's benign cause — an output-budget death or a
    candidate-process crash score the tail as errors, while a clean exit that
    simply reported fewer cases is incomplete coverage, not a fault.

    A delivered item whose payload the case schema rejects is a harness
    failure the raising lane surfaces; the metrics lane catches it and scores
    it as candidate case data.
    """
    run = result.run
    attribution = run.outcome.attribution

    if _is_wall_clock_budget(run):
        return timeout_results(
            task=task,
            function_name=function_name,
            timeout_seconds=timeout_seconds,
        )

    if attribution not in (Attribution.PAYLOAD, Attribution.BUDGET):
        raise EvaluationHarnessError(
            f"runner batch failed with {attribution.value} attribution",
            case_results=error_results(
                task=task,
                function_name=function_name,
                message=(
                    f"runner batch failed with {attribution.value} attribution"
                ),
            ),
        )

    results = _delivered_case_results(
        task=task,
        function_name=function_name,
        result=result,
    )
    synthesized_status, synthesized_message = _synthesized_disposition(run)
    if synthesized_status is not None:
        for case_id in result.missing_item_ids:
            results.append(
                _case_result(
                    task=task,
                    function_name=function_name,
                    case_id=case_id,
                    status=synthesized_status,
                    message=synthesized_message,
                )
            )
    # A clean exit that reported fewer cases is incomplete coverage: the
    # missing cases are left absent, and coverage_complete reads False.
    return results


def _delivered_case_results(
    *,
    task: HumanEvalTask,
    function_name: str,
    result: BatchResult,
) -> list[EvaluationCaseResult]:
    parsed_tests = require_parsed_tests(task)
    adapter = TypeAdapter(HumanEvalRunnerCaseOutput)
    results: list[EvaluationCaseResult] = []
    for item in result.results:
        results.append(
            _case_from_item(
                task=task,
                function_name=function_name,
                parsed_tests=parsed_tests,
                item=item,
                adapter=adapter,
            )
        )
    return results


def _case_from_item(
    *,
    task: HumanEvalTask,
    function_name: str,
    parsed_tests: ParsedTests,
    item: ItemResult,
    adapter: TypeAdapter[HumanEvalRunnerCaseOutput],
) -> EvaluationCaseResult:
    # A body that raised arrives as a kit error payload: the candidate's own
    # traceback, preserved as an ERROR case scored against the candidate.
    error_text = item.error_text
    if error_text is not None:
        return _case_result(
            task=task,
            function_name=function_name,
            case_id=item.item_id,
            status=EvaluationCaseStatus.ERROR,
            message=error_text,
            metadata=_case_metadata(parsed_tests, item.item_id),
        )
    payload = dict(item.payload) if isinstance(item.payload, dict) else {}
    payload["case_id"] = item.item_id
    try:
        parsed = adapter.validate_python(payload)
    except ValidationError as exc:
        error = _case_result(
            task=task,
            function_name=function_name,
            case_id=item.item_id,
            status=EvaluationCaseStatus.ERROR,
            message=f"Invalid runner output: {exc}",
            metadata=_case_metadata(parsed_tests, item.item_id),
        )
        raise EvaluationHarnessError(
            "runner output case failed validation",
            case_results=[error],
            cause=exc,
        ) from exc
    return EvaluationCaseResult(
        task_id=task.task_id,
        case_id=parsed.case_id,
        function_name=function_name,
        status=parsed.status,
        message=parsed.message,
        test_type=parsed_tests.test_type,
        input_repr=parsed.input_repr,
        expected_output_repr=parsed.expected_output_repr,
        actual_output_repr=parsed.actual_output_repr,
        elapsed_seconds=parsed.elapsed_seconds,
        timeout_seconds=parsed.timeout_seconds,
    )


def _synthesized_disposition(
    run: RunResult,
) -> tuple[EvaluationCaseStatus | None, str]:
    """How an unreported case is scored, from the run's outcome.

    Returns ``(None, "")`` when the run itself carries no benign cause for a
    missing case — that absence is a harness fault the caller raises on.
    """
    attribution = run.outcome.attribution
    if (
        attribution is Attribution.BUDGET
        and run.outcome.violated_axis is BudgetAxis.OUTPUT
    ):
        return (
            EvaluationCaseStatus.ERROR,
            "subprocess output budget exceeded before this case reported",
        )
    if run.returncode in CANDIDATE_KILL_RETURNCODES:
        message = (
            "subprocess killed candidate execution "
            f"(exit {run.returncode}: external kill or interpreter crash)"
        )
        detail = run.stderr.strip()
        if detail:
            message = f"{message}: {detail}"
        return (EvaluationCaseStatus.ERROR, message)
    return (None, "")


def _is_wall_clock_budget(run: RunResult) -> bool:
    return (
        run.outcome.attribution is Attribution.BUDGET
        and run.outcome.violated_axis is BudgetAxis.WALL_CLOCK
    )


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


def _case_result(
    *,
    task: HumanEvalTask,
    function_name: str,
    case_id: str,
    status: EvaluationCaseStatus,
    message: str,
    metadata: dict[str, str] | None = None,
) -> EvaluationCaseResult:
    parsed_tests = require_parsed_tests(task)
    resolved = (
        metadata
        if metadata is not None
        else _case_metadata(parsed_tests, case_id)
    )
    return EvaluationCaseResult(
        task_id=task.task_id,
        case_id=case_id,
        function_name=function_name,
        status=status,
        message=message,
        test_type=parsed_tests.test_type,
        input_repr=resolved.get("input_repr", ""),
        expected_output_repr=resolved.get("expected_output_repr", ""),
        actual_output_repr=resolved.get("actual_output_repr", ""),
    )


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


def _case_metadata(parsed_tests: ParsedTests, case_id: str) -> dict[str, str]:
    for case in parsed_tests.cases:
        if case.case_id == case_id:
            return case_metadata(parsed_tests, case)
    return {
        "input_repr": "",
        "expected_output_repr": "",
        "actual_output_repr": "",
    }


def require_parsed_tests(task: HumanEvalTask) -> ParsedTests:
    if task.parsed_tests is None:
        raise ValueError("HumanEvalTask.parsed_tests is required")
    return task.parsed_tests


@cache
def driver_body_template() -> str:
    # Read as a resource rather than imported: the body has module-level side
    # effects when composed and must remain dependency-free.
    return (
        files("dr_code.humaneval")
        .joinpath("batch_runner_script.py")
        .read_text(encoding="utf-8")
    )
