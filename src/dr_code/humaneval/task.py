"""HumanEval task, parsing, and subprocess execution primitives.

The subprocess runner validates each returned case result, but it currently
preserves partial runner output rather than requiring one returned row per
parsed test case. Tightening that cardinality check would be a benchmark
behavior change and is deferred until per-test score persistence semantics are
defined. Returned case ids must still be known and unique so partial output
can never inflate coverage.

Failure attribution: candidate-attributable terminations (memory/CPU-limit
SIGKILL, interpreter crash, SystemExit, output floods) are scored as case
errors or timeouts; ``EvaluationHarnessError``/``HarnessFailure`` is reserved
for sandbox or runtime breakage so operators can alert on it. Candidate code
runs in the same in-container interpreter as the trusted runner, so a
deliberately adversarial candidate can still forge its own task's case
results; the sandbox boundary guarantees host, credential, and network
isolation, not single-task score integrity against adversarial submissions.
"""

from __future__ import annotations

import ast
import json
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from importlib.resources import files
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    computed_field,
    model_validator,
)

from dr_code.humaneval.parsed_code import ParsedCode, parse_code
from dr_code.humaneval.parsed_tests import (
    HumanEvalTestCaseKind,
    InputExpressionTestCase,
    InputOracleTestCase,
    InputResultTestCase,
    ParsedTests,
    SingleCaseCheck,
    TestCase,
    UnsupportedTestFormatError,
    assertion_tolerance,
    find_assert_statement,
    find_assertion_call,
    find_assignment_value,
    find_for_loop,
    find_oracle_name,
    for_loop_names,
    literal_assignment,
)
from dr_code.humaneval.sandbox import (
    CANDIDATE_KILL_RETURNCODES,
    SandboxOutputLimitError,
    SandboxRunner,
    SandboxTimeoutError,
    run_python_in_sandbox,
)


class EvaluationCaseStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"


class HumanEvalTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    prompt: str
    canonical_solution: str
    entry_point: str
    test: str
    notes: list[str] = Field(default_factory=list)
    parsed: ParsedCode | None = None
    parsed_tests: ParsedTests | None = None

    @computed_field
    @property
    def ground_truth_code(self) -> str:
        return self.prompt + self.canonical_solution

    @computed_field
    @property
    def ground_truth_code_without_comments(self) -> str | None:
        if self.parsed is None:
            return None
        return self.parsed.code_without_comments

    @model_validator(mode="after")
    def parse_code(self) -> Self:
        if self.parsed is None:
            self.parsed = parse_code(
                display_title=self.task_id,
                code_str=self.ground_truth_code,
            )
        if self.parsed_tests is None:
            self.parsed_tests = parse_human_eval_tests(self.test)
        return self


@dataclass(frozen=True, slots=True)
class EvaluationCaseResult:
    task_id: str
    case_id: str
    function_name: str
    status: EvaluationCaseStatus
    test_type: HumanEvalTestCaseKind
    message: str = ""
    input_repr: str = ""
    expected_output_repr: str = ""
    actual_output_repr: str = ""
    elapsed_seconds: float | None = None
    timeout_seconds: float | None = None

    def to_summary(self) -> EvaluationCaseSummary:
        return EvaluationCaseSummary(
            task_id=self.task_id,
            case_id=self.case_id,
            function_name=self.function_name,
            status=self.status,
            message=self.message,
            test_type=self.test_type,
            input_repr=self.input_repr,
            expected_output_repr=self.expected_output_repr,
            actual_output_repr=self.actual_output_repr,
            elapsed_seconds=self.elapsed_seconds,
            timeout_seconds=self.timeout_seconds,
        )


class EvaluationCaseSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    case_id: str
    function_name: str
    status: EvaluationCaseStatus
    message: str = ""
    test_type: HumanEvalTestCaseKind
    input_repr: str = ""
    expected_output_repr: str = ""
    actual_output_repr: str = ""
    elapsed_seconds: float | None = None
    timeout_seconds: float | None = None


def _results_for_function(
    results: list[EvaluationCaseResult],
    function_name: str,
) -> list[EvaluationCaseResult]:
    return [
        result for result in results if result.function_name == function_name
    ]


def select_best_function_name(
    *,
    function_names: list[str],
    entry_point: str,
    results: list[EvaluationCaseResult],
) -> str | None:
    if not function_names:
        return None
    return max(
        function_names,
        key=lambda function_name: (
            sum(
                1
                for result in results
                if result.function_name == function_name
                and result.status is EvaluationCaseStatus.PASSED
            ),
            function_name == entry_point,
            -function_names.index(function_name),
        ),
    )


class EvaluationTaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    entry_point: str
    function_names: list[str]
    total_cases: int
    results: list[EvaluationCaseResult] = Field(default_factory=list)

    @computed_field
    @property
    def best_function_name(self) -> str | None:
        return select_best_function_name(
            function_names=self.function_names,
            entry_point=self.entry_point,
            results=self.results,
        )

    @computed_field
    @property
    def failures(self) -> list[EvaluationCaseResult]:
        best_function_name = self.best_function_name
        if best_function_name is None:
            return []
        return [
            result
            for result in self.results
            if result.function_name == best_function_name
            and result.status is not EvaluationCaseStatus.PASSED
        ]

    @computed_field
    @property
    def coverage_complete(self) -> bool:
        best_function_name = self.best_function_name
        if best_function_name is None:
            return False
        function_results = _results_for_function(
            self.results,
            best_function_name,
        )
        return len(function_results) == self.total_cases

    @computed_field
    @property
    def passed(self) -> bool:
        best_function_name = self.best_function_name
        if best_function_name is None:
            return False
        function_results = _results_for_function(
            self.results,
            best_function_name,
        )
        if not self.coverage_complete:
            return False
        return all(
            result.status is EvaluationCaseStatus.PASSED
            for result in function_results
        )

    @computed_field
    @property
    def status_counts(self) -> dict[str, int]:
        best_function_name = self.best_function_name
        if best_function_name is None:
            return {}
        return dict(
            Counter(
                result.status.value
                for result in self.results
                if result.function_name == best_function_name
            )
        )

    def to_summary(self) -> EvaluationTaskSummary:
        return EvaluationTaskSummary(
            task_id=self.task_id,
            entry_point=self.entry_point,
            function_names=self.function_names,
            best_function_name=self.best_function_name,
            total_cases=self.total_cases,
            results=[result.to_summary() for result in self.results],
            passed=self.passed,
            failure_count=len(self.failures),
            status_counts=self.status_counts,
        )


class EvaluationTaskSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    entry_point: str
    function_names: list[str]
    best_function_name: str | None = None
    total_cases: int
    results: list[EvaluationCaseSummary] = Field(default_factory=list)
    passed: bool
    failure_count: int
    status_counts: dict[str, int] = Field(default_factory=dict)


class HumanEvalRunnerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    candidate_code: str
    support_code: str
    function_name: str
    test_type: HumanEvalTestCaseKind
    checks: list[SingleCaseCheck]


class HumanEvalRunnerCaseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    status: EvaluationCaseStatus
    message: str = ""
    input_repr: str = ""
    expected_output_repr: str = ""
    actual_output_repr: str = ""
    elapsed_seconds: float | None = None
    timeout_seconds: float | None = None


class EvaluationHarnessError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        case_results: Iterable[EvaluationCaseResult] = (),
        evaluation: EvaluationTaskResult | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.case_results = list(case_results)
        self.evaluation = evaluation
        self.cause = cause


class HumanEvalOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")

    notes: list[str] = Field(default_factory=list)
    canonical_solution: str | None = None
    test_replacements: dict[str, str] = Field(default_factory=dict)


HUMAN_EVAL_OVERRIDES: dict[str, HumanEvalOverride] = {
    "HumanEval/32": HumanEvalOverride(
        notes=[
            "Fixed the benchmark test assertion to evaluate the polynomial at "
            "the returned root with a scaled residual tolerance, and replaced "
            "the Newton-only canonical solution with a hybrid "
            "Newton/bisection method."
        ],
        canonical_solution="""

    dxs = [xs[i] * i for i in range(1, len(xs))]

    def func(x):
        return poly(xs, x)

    def derivative(x):
        return poly(dxs, x)

    x = 0.0
    last_step = None
    for _ in range(1000):
        fx = func(x)
        dfx = derivative(x)
        if abs(fx) < 1e-5:
            return x
        if dfx == 0:
            break
        last_step = fx / dfx
        x = x - last_step

    if last_step is not None and abs(last_step) <= 1e-7 * max(1.0, abs(x)):
        return x

    lo, hi = -1.0, 1.0
    flo, fhi = func(lo), func(hi)
    for _ in range(200):
        if flo == 0:
            return lo
        if fhi == 0:
            return hi
        if (flo < 0 < fhi) or (fhi < 0 < flo):
            break
        lo *= 2.0
        hi *= 2.0
        flo, fhi = func(lo), func(hi)

    for _ in range(200):
        mid = (lo + hi) / 2.0
        fm = func(mid)
        if fm == 0 or abs(hi - lo) <= 1e-12 * max(1.0, abs(mid)):
            return mid
        if (flo < 0 < fm) or (fm < 0 < flo):
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm

    return (lo + hi) / 2.0
""",
        test_replacements={
            "assert _poly(*candidate(*inp), inp) <= 0.0001": (
                "assert abs(_poly(*inp, (out := candidate(*inp)))) <= max("
                "1e-4, "
                "1e-12 * sum("
                "abs(coeff) * max(1.0, abs(out)) ** j "
                "for j, coeff in enumerate(inp[0])"
                ")"
                ")"
            ),
        },
    ),
}


def parse_human_eval_tests(test_str: str) -> ParsedTests:
    tree = ast.parse(test_str)
    check_node = find_check_function(tree)
    if len(check_node.args.args) != 1:
        raise UnsupportedTestFormatError(
            "Expected check(candidate) with one positional argument"
        )

    inputs = literal_assignment(check_node, "inputs")
    results_value = find_assignment_value(check_node, "results")
    assertion_call = find_assertion_call(check_node)
    tolerance = assertion_tolerance(assertion_call) if assertion_call else 0
    support_code = support_code_without_check(tree)
    candidate_arg_name = check_node.args.args[0].arg

    cases: list[TestCase]
    if results_value is not None:
        results = literal_assignment(check_node, "results")
        if len(inputs) != len(results):
            raise UnsupportedTestFormatError(
                f"len(inputs)={len(inputs)} does not match "
                f"len(results)={len(results)}"
            )
        if assertion_call is None:
            loop_node = find_for_loop(check_node)
            index_name, input_name, expected_name = for_loop_names(loop_node)
            assert_statement = find_assert_statement(check_node)
            cases = [
                InputExpressionTestCase(
                    case_id=f"case_{index}",
                    args=args,
                    expected=expected,
                    expression=ast.unparse(assert_statement),
                    input_name=input_name,
                    expected_name=expected_name,
                    index_name=index_name,
                )
                for index, (args, expected) in enumerate(
                    zip(inputs, results, strict=True)
                )
            ]
            test_type = HumanEvalTestCaseKind.INPUT_EXPRESSION
        else:
            cases = [
                InputResultTestCase(
                    case_id=f"case_{index}",
                    args=args,
                    expected=expected,
                    atol=tolerance,
                )
                for index, (args, expected) in enumerate(
                    zip(inputs, results, strict=True)
                )
            ]
            test_type = HumanEvalTestCaseKind.INPUT_RESULT
    else:
        _ = find_for_loop(check_node)
        if assertion_call is None:
            raise UnsupportedTestFormatError(
                "Expected assertion(..., ref_func(*inp), ...) for oracle tests"
            )
        oracle_name = find_oracle_name(assertion_call)
        if oracle_name is None:
            raise UnsupportedTestFormatError(
                "Expected assertion(..., ref_func(*inp), ...) for oracle tests"
            )
        cases = [
            InputOracleTestCase(
                case_id=f"case_{index}",
                args=args,
                oracle_name=oracle_name,
                atol=tolerance,
            )
            for index, args in enumerate(inputs)
        ]
        test_type = HumanEvalTestCaseKind.INPUT_ORACLE

    return ParsedTests(
        test_type=test_type,
        support_code=support_code,
        check_name=check_node.name,
        candidate_arg_name=candidate_arg_name,
        assertion_name="assertion",
        cases=cases,
        original_test=test_str,
    )


def parse_human_eval_dataset(
    rows: Iterable[Mapping[str, Any]],
    *,
    overrides: dict[str, HumanEvalOverride] | None = None,
) -> list[HumanEvalTask]:
    active_overrides = HUMAN_EVAL_OVERRIDES if overrides is None else overrides
    return [
        HumanEvalTask(**apply_human_eval_override(row, active_overrides))
        for row in rows
    ]


def apply_human_eval_override(
    row: Mapping[str, Any],
    overrides: dict[str, HumanEvalOverride],
) -> dict[str, Any]:
    task_id = str(row["task_id"])
    override = overrides.get(task_id)
    if override is None:
        return dict(row)

    updated = dict(row)
    if override.canonical_solution is not None:
        updated["canonical_solution"] = override.canonical_solution
    test = str(updated["test"])
    for old, new in override.test_replacements.items():
        if old not in test:
            raise ValueError(
                f"Override replacement text not found for {task_id}"
            )
        test = test.replace(old, new, 1)
    updated["test"] = test
    updated["notes"] = [*updated.get("notes", []), *override.notes]
    return updated


def evaluate_human_eval_code(
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


def find_check_function(tree: ast.Module) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            return node
    raise UnsupportedTestFormatError(
        "Could not find check(candidate) function"
    )


def support_code_without_check(tree: ast.Module) -> str:
    support_nodes = [
        node
        for node in tree.body
        if not (isinstance(node, ast.FunctionDef) and node.name == "check")
    ]
    module = ast.Module(body=support_nodes, type_ignores=[])
    return ast.unparse(module)


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
    run_in_sandbox: SandboxRunner = run_python_in_sandbox,
) -> list[EvaluationCaseResult]:
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
    started_at = time.perf_counter()
    try:
        completed = run_in_sandbox(
            source=runner_source or runner_script(),
            input_json=payload.model_dump_json(),
            timeout_seconds=timeout_seconds,
        )
    except SandboxTimeoutError:
        return timeout_results(
            task=task,
            function_name=function_name,
            timeout_seconds=timeout_seconds,
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
    elapsed_seconds = time.perf_counter() - started_at

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
