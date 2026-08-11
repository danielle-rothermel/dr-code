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
from pydantic import Field, model_validator

from dr_code.core.models import FrozenModel
from dr_code.evaluation.identity import (
    EvaluationCandidateIdentity,
    MaterializedEvaluationCandidate,
)
from dr_code.humaneval.settings import CodeTestSettings
from dr_code.humaneval.task import EvaluationCaseResult, HumanEvalTask
from dr_code.humaneval.task import EvaluationCaseStatus
from dr_code.humaneval.parsed_tests import ParsedTests, SingleCaseCheck
from dr_code.metrics import MetricName, MetricQuestionCoordinate
from dr_code.metrics.coordinates import question_settings

HUMANEVAL_CANDIDATE_JOB_SCHEMA_VERSION: Final = 1
HUMANEVAL_CANDIDATE_ENTRY_POINT: Final = ImportableEntryPoint(
    module_name="dr_code.humaneval.job",
    attribute_name="evaluate_humaneval_candidate_job",
)


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
    schema_version: Literal[1] = HUMANEVAL_CANDIDATE_JOB_SCHEMA_VERSION
    candidate: MaterializedEvaluationCandidate
    suites: tuple[HumanEvalEvaluatorSuite, ...] = Field(min_length=1)


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
    cases: tuple[EvaluationCaseResult, ...]


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
    schema_version: Literal[1] = HUMANEVAL_CANDIDATE_JOB_SCHEMA_VERSION
    candidate: EvaluationCandidateIdentity
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


_FIELD_LIMIT: Final = 8_000


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
                message=_exception_message(error),
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
            _evaluate_suite(suite, namespace, candidate_functions)
            for suite in validated.suites
        ),
    )
    return result.model_dump(mode="json", exclude_computed_fields=True)


def _evaluate_suite(
    suite: HumanEvalEvaluatorSuite,
    candidate_namespace: dict[str, object],
    candidate_functions: Mapping[str, object],
) -> HumanEvalSuiteResult:
    parsed_tests = suite.task.parsed_tests
    if parsed_tests is None:
        return HumanEvalSuiteHarnessFailure(
            question=suite.question,
            failure_type="ValueError",
            message="HumanEvalTask.parsed_tests is required",
            completed_groups=(),
        )

    try:
        exec(parsed_tests.support_code, candidate_namespace)
        compiled_checks = tuple(
            (
                check,
                compile(
                    check.code,
                    f"<generated {check.case_id}>",
                    "exec",
                ),
            )
            for check in parsed_tests.iter_checks(candidate_name="candidate")
        )
    except BaseException as error:
        return HumanEvalSuiteHarnessFailure(
            question=suite.question,
            failure_type=type(error).__name__,
            message=_exception_message(error),
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
                    )
                    for check, compiled_check in compiled_checks
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
) -> EvaluationCaseResult:
    started_at = time.perf_counter()
    check_namespace = candidate_namespace | {"candidate": candidate}
    try:
        exec(compiled_check, check_namespace)
    except AssertionError as error:
        status = EvaluationCaseStatus.FAILED
        message = _clip(error)
        metadata = _failure_metadata(
            check,
            candidate,
            candidate_namespace,
        )
    except BaseException as error:
        status = EvaluationCaseStatus.ERROR
        message = _exception_message(error)
        metadata = _failure_metadata(
            check,
            candidate,
            candidate_namespace,
        )
    else:
        status = EvaluationCaseStatus.PASSED
        message = ""
        metadata = {
            "input_repr": check.input_repr,
            "expected_output_repr": check.expected_output_repr,
            "actual_output_repr": "",
        }
    return EvaluationCaseResult(
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
) -> dict[str, str]:
    namespace = candidate_namespace | {"candidate": candidate}
    actual = ""
    expected = check.expected_output_repr
    try:
        if check.actual_output_expr:
            actual = _clip(eval(check.actual_output_expr, namespace))
    except BaseException as error:
        actual = _exception_message(error)
    try:
        if check.expected_output_expr:
            expected = _clip(eval(check.expected_output_expr, namespace))
    except BaseException as error:
        expected = _exception_message(error)
    return {
        "input_repr": check.input_repr,
        "expected_output_repr": expected,
        "actual_output_repr": actual,
    }


def _assertion(actual: Any, expected: Any, atol: float = 0) -> None:
    if atol:
        assert abs(actual - expected) <= atol
    else:
        assert actual == expected


def _clip(value: object) -> str:
    text = str(value)
    if len(text) > _FIELD_LIMIT:
        return text[:_FIELD_LIMIT] + "...[truncated]"
    return text


def _exception_message(error: BaseException) -> str:
    rendered = "".join(
        traceback.format_exception_only(type(error), error)
    ).strip()
    return _clip(rendered)


__all__ = [
    "CandidateNamespaceFailure",
    "CandidateNamespaceLoaded",
    "CandidateNamespaceOutcome",
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
