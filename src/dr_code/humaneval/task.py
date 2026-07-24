"""HumanEval task models, benchmark overrides, and dataset parsing.

Holds the serialization-boundary task and summary models, frozen internal
case results, best-function selection, and dataset parsing. Test parsing lives
in ``parsed_tests``; execution orchestration and its standalone resource live
in ``batch_runner``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

from dr_code.humaneval.parsed_code import ParsedCode, parse_code
from dr_code.humaneval.parsed_tests import (
    HumanEvalTestCaseKind,
    ParsedTests,
    SingleCaseCheck,
    parse_human_eval_tests,
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
