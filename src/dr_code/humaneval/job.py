from __future__ import annotations

import ast
import inspect
import time
import traceback
from collections.abc import Mapping
from types import CodeType
from typing import Annotated, Any, Final, Literal, Self, TypeAlias

from dr_serialize import Jsonable
from dr_exec import ImportableEntryPoint
from pydantic import Field, PositiveInt, model_validator

from dr_code.core.models import FrozenModel
from dr_code.evaluation.identity import (
    EvalCandidateIdentity,
    MaterializedEvalCandidate,
)
from dr_code.humaneval.settings import CodeTestSettings
from dr_code.humaneval.task import EvalCaseResult, HumanEvalTask
from dr_code.humaneval.task import EvalCaseStatus
from dr_code.humaneval.parsed_tests import (
    EXPECTED_OUTPUT_NAME as _EXPECTED_OUTPUT_NAME,
)
from dr_code.humaneval.parsed_tests import ParsedTests, SingleCaseCheck
from dr_code.metrics import MetricName, MetricQuestionCoordinate
from dr_code.metrics.coordinates import question_settings

HUMANEVAL_CANDIDATE_JOB_SCHEMA_VERSION: Final = 2
HUMANEVAL_CANDIDATE_ENTRY_POINT: Final = ImportableEntryPoint(
    module_name="dr_code.humaneval.job",
    attribute_name="evaluate_humaneval_candidate_job",
)
DEFAULT_FIELD_LIMIT: Final = 32_000
FIELD_TRUNCATION_MARKER: Final = "...[truncated]"


class HumanEvalEvaluatorSuite(FrozenModel):
    question: MetricQuestionCoordinate
    task: HumanEvalTask
    settings: CodeTestSettings

    @model_validator(mode="after")
    def validate_question(self) -> Self:
        if self.question.metric is not MetricName.CODE_TEST:
            raise ValueError("a HumanEval evaluator suite requires code_test")
        if self.question.settings != question_settings(self.settings):
            raise ValueError(
                "suite question settings must equal its CodeTest settings"
            )
        return self


class HumanEvalCandidateJobRequest(FrozenModel):
    schema_version: Literal[2] = HUMANEVAL_CANDIDATE_JOB_SCHEMA_VERSION
    candidate: MaterializedEvalCandidate
    suites: tuple[HumanEvalEvaluatorSuite, ...] = Field(min_length=1)
    # Bounds every rendered evidence field this job reports. The default
    # holds a realistic failing assertion's repr rather than cutting it at
    # the point a reader needs.
    field_limit: PositiveInt = DEFAULT_FIELD_LIMIT


class CandidateNamespaceLoaded(FrozenModel):
    kind: Literal["loaded"] = "loaded"
    function_names: tuple[str, ...]


class CandidateNamespaceFailure(FrozenModel):
    kind: Literal["candidate_failure"] = "candidate_failure"
    failure_type: str
    message: str


CandidateNamespaceOutcome: TypeAlias = Annotated[
    CandidateNamespaceLoaded | CandidateNamespaceFailure,
    Field(discriminator="kind"),
]


class HumanEvalFunctionGroupResult(FrozenModel):
    function_name: str
    cases: tuple[EvalCaseResult, ...]


class HumanEvalSuiteCompleted(FrozenModel):
    kind: Literal["completed"] = "completed"
    question: MetricQuestionCoordinate
    groups: tuple[HumanEvalFunctionGroupResult, ...]


class HumanEvalSuiteHarnessFailure(FrozenModel):
    kind: Literal["harness_failure"] = "harness_failure"
    question: MetricQuestionCoordinate
    failure_type: str
    message: str
    completed_groups: tuple[HumanEvalFunctionGroupResult, ...]


HumanEvalSuiteResult: TypeAlias = Annotated[
    HumanEvalSuiteCompleted | HumanEvalSuiteHarnessFailure,
    Field(discriminator="kind"),
]


class HumanEvalCandidateJobResult(FrozenModel):
    schema_version: Literal[2] = HUMANEVAL_CANDIDATE_JOB_SCHEMA_VERSION
    candidate: EvalCandidateIdentity
    namespace: CandidateNamespaceOutcome
    suites: tuple[HumanEvalSuiteResult, ...]

    @model_validator(mode="after")
    def validate_namespace_result(self) -> Self:
        if isinstance(self.namespace, CandidateNamespaceFailure):
            if self.suites:
                raise ValueError(
                    "a namespace-loading failure cannot carry suite results"
                )
        elif not self.suites:
            raise ValueError(
                "a loaded namespace must carry the requested suite results"
            )
        return self


def evaluate_humaneval_candidate_job(request: Jsonable) -> Jsonable:
    """Evaluate every requested suite after loading one candidate namespace."""

    validated = HumanEvalCandidateJobRequest.model_validate(request)
    namespace: dict[str, object] = {"assertion": _assertion}
    source = validated.candidate.source.source
    try:
        tree = ast.parse(source)
        candidate_code = compile(tree, "<evaluation candidate>", "exec")
        exec(candidate_code, namespace)
    except BaseException as error:
        result = HumanEvalCandidateJobResult(
            candidate=validated.candidate.identity,
            namespace=CandidateNamespaceFailure(
                failure_type=type(error).__name__,
                message=_exception_message(error, validated.field_limit),
            ),
            suites=(),
        )
        return result.model_dump(mode="json", exclude_computed_fields=True)

    declared_function_names = dict.fromkeys(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    )
    candidate_functions = {
        name: value
        for name in declared_function_names
        if inspect.isfunction(value := namespace.get(name))
    }
    function_names = tuple(candidate_functions)
    result = HumanEvalCandidateJobResult(
        candidate=validated.candidate.identity,
        namespace=CandidateNamespaceLoaded(function_names=function_names),
        suites=tuple(
            _evaluate_suite(
                suite,
                namespace,
                candidate_functions,
                field_limit=validated.field_limit,
            )
            for suite in validated.suites
        ),
    )
    return result.model_dump(mode="json", exclude_computed_fields=True)


def _evaluate_suite(
    suite: HumanEvalEvaluatorSuite,
    candidate_namespace: dict[str, object],
    candidate_functions: Mapping[str, object],
    *,
    field_limit: int,
) -> HumanEvalSuiteResult:
    parsed_tests = suite.task.parsed_tests

    try:
        exec(parsed_tests.support_code, candidate_namespace)
        # The oracle is trusted test code, so evaluate it here rather than
        # inside the candidate's case block: a failing oracle is a harness
        # failure, never a candidate failure.
        compiled_checks = tuple(
            (
                check,
                compile(
                    check.code,
                    f"<generated {check.case_id}>",
                    "exec",
                ),
                None
                if check.expected_output_expr is None
                else eval(check.expected_output_expr, candidate_namespace),
            )
            for check in parsed_tests.iter_checks(candidate_name="candidate")
        )
    except BaseException as error:
        return HumanEvalSuiteHarnessFailure(
            question=suite.question,
            failure_type=type(error).__name__,
            message=_exception_message(error, field_limit),
            completed_groups=(),
        )

    groups: list[HumanEvalFunctionGroupResult] = []
    for function_name, candidate in candidate_functions.items():
        groups.append(
            HumanEvalFunctionGroupResult(
                function_name=function_name,
                cases=tuple(
                    _evaluate_case(
                        task=suite.task,
                        parsed_tests=parsed_tests,
                        function_name=function_name,
                        candidate=candidate,
                        candidate_namespace=candidate_namespace,
                        check=check,
                        compiled_check=compiled_check,
                        expected_output=expected_output,
                        field_limit=field_limit,
                    )
                    for check, compiled_check, expected_output in (
                        compiled_checks
                    )
                ),
            )
        )
    return HumanEvalSuiteCompleted(
        question=suite.question,
        groups=tuple(groups),
    )


def _evaluate_case(
    *,
    task: HumanEvalTask,
    parsed_tests: ParsedTests,
    function_name: str,
    candidate: object,
    candidate_namespace: dict[str, object],
    check: SingleCaseCheck,
    compiled_check: CodeType,
    expected_output: object = None,
    field_limit: int,
) -> EvalCaseResult:
    started_at = time.perf_counter()
    check_namespace = candidate_namespace | {"candidate": candidate}
    if check.expected_output_expr is not None:
        check_namespace[_EXPECTED_OUTPUT_NAME] = expected_output
    try:
        exec(compiled_check, check_namespace)
    except AssertionError as error:
        status = EvalCaseStatus.FAILED
        message = _clip(error, field_limit)
        metadata = _failure_metadata(
            check,
            candidate,
            candidate_namespace,
            expected_output,
            field_limit,
        )
    except BaseException as error:
        status = EvalCaseStatus.ERROR
        message = _exception_message(error, field_limit)
        metadata = _failure_metadata(
            check,
            candidate,
            candidate_namespace,
            expected_output,
            field_limit,
        )
    else:
        status = EvalCaseStatus.PASSED
        message = ""
        metadata = {
            "input_repr": _clip(check.input_repr, field_limit),
            "expected_output_repr": _clip(
                check.expected_output_repr, field_limit
            ),
            "actual_output_repr": "",
        }
    return EvalCaseResult(
        task_id=task.task_id,
        case_id=check.case_id,
        function_name=function_name,
        status=status,
        message=message,
        test_type=parsed_tests.test_type,
        input_repr=metadata["input_repr"],
        expected_output_repr=metadata["expected_output_repr"],
        actual_output_repr=metadata["actual_output_repr"],
        elapsed_seconds=time.perf_counter() - started_at,
    )


def _failure_metadata(
    check: SingleCaseCheck,
    candidate: object,
    candidate_namespace: dict[str, object],
    expected_output: object,
    field_limit: int,
) -> dict[str, str]:
    namespace = candidate_namespace | {"candidate": candidate}
    actual = ""
    expected = check.expected_output_repr
    try:
        if check.actual_output_expr:
            actual = _clip(
                eval(check.actual_output_expr, namespace), field_limit
            )
    except BaseException as error:
        actual = _exception_message(error, field_limit)
    # The oracle already ran as trusted code, so report its value rather than
    # re-evaluating it here.
    if check.expected_output_expr is not None:
        expected = _clip(expected_output, field_limit)
    else:
        expected = _clip(expected, field_limit)
    return {
        "input_repr": _clip(check.input_repr, field_limit),
        "expected_output_repr": expected,
        "actual_output_repr": actual,
    }


def _assertion(actual: Any, expected: Any, atol: float = 0) -> None:
    if atol:
        assert abs(actual - expected) <= atol
    else:
        assert actual == expected


def _clip(value: object, field_limit: int) -> str:
    text = str(value)
    if len(text) > field_limit:
        return text[:field_limit] + FIELD_TRUNCATION_MARKER
    return text


def _exception_message(error: BaseException, field_limit: int) -> str:
    rendered = "".join(
        traceback.format_exception_only(type(error), error)
    ).strip()
    return _clip(rendered, field_limit)


__all__ = [
    "CandidateNamespaceFailure",
    "CandidateNamespaceLoaded",
    "CandidateNamespaceOutcome",
    "DEFAULT_FIELD_LIMIT",
    "FIELD_TRUNCATION_MARKER",
    "HUMANEVAL_CANDIDATE_ENTRY_POINT",
    "HUMANEVAL_CANDIDATE_JOB_SCHEMA_VERSION",
    "HumanEvalCandidateJobRequest",
    "HumanEvalCandidateJobResult",
    "HumanEvalEvaluatorSuite",
    "HumanEvalFunctionGroupResult",
    "HumanEvalSuiteCompleted",
    "HumanEvalSuiteHarnessFailure",
    "HumanEvalSuiteResult",
    "evaluate_humaneval_candidate_job",
]
