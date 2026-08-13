from __future__ import annotations

import ast
import inspect
from typing import Annotated, Final, Literal, TypeAlias, cast

from dr_exec import ImportableEntryPoint
from dr_serialize import Jsonable
from pydantic import Field

from dr_code.core.models import FrozenModel
from dr_code.evaluation.candidate_job import (
    CandidateEvaluatorSuite,
    CandidateJobResult,
)
from dr_code.evaluation.id import (
    EvalCandidateId,
    MaterializedEvalCandidate,
)
from dr_code.metrics.coordinates import MetricQuestionCoordinate

STUB_CANDIDATE_JOB_SCHEMA_VERSION: Final = 1
STUB_REQUEST_IDENTITY_SCHEMA: Final = "dr-code/stub-candidate-job-v1"
STUB_CANDIDATE_ENTRY_POINT: Final = ImportableEntryPoint(
    module_name="_stubs.candidate_harness.job",
    attribute_name="evaluate_stub_candidate_job",
)
DEFAULT_FIELD_LIMIT: Final = 32_000


class StubEvaluatorSuite(FrozenModel):
    question: MetricQuestionCoordinate
    entry_point: str
    inputs: tuple[tuple[object, ...], ...]
    expected: tuple[object, ...]
    support_failure: bool = False


class StubCandidateJobRequest(FrozenModel):
    schema_version: Literal[1] = STUB_CANDIDATE_JOB_SCHEMA_VERSION
    candidate: MaterializedEvalCandidate
    suites: tuple[StubEvaluatorSuite, ...] = Field(min_length=1)
    field_limit: int = DEFAULT_FIELD_LIMIT

    def execution_entry_point(self) -> ImportableEntryPoint:
        return STUB_CANDIDATE_ENTRY_POINT

    def request_identity_schema(self) -> str:
        return STUB_REQUEST_IDENTITY_SCHEMA

    def request_identity_schema_version(self) -> int:
        return STUB_CANDIDATE_JOB_SCHEMA_VERSION

    def request_payload(self) -> Jsonable:
        return cast(
            Jsonable,
            self.model_dump(mode="json", exclude_computed_fields=True),
        )

    def result_model(self) -> type[StubCandidateJobResult]:
        return StubCandidateJobResult

    def validate_result_matches_request(
        self,
        result: CandidateJobResult,
        /,
    ) -> None:
        if not isinstance(result, StubCandidateJobResult):
            raise TypeError("result must be a StubCandidateJobResult")
        if result.candidate != self.candidate.identity:
            raise ValueError(
                "candidate job result identity does not match request"
            )
        questions = tuple(suite.question for suite in self.suites)
        result_questions = tuple(suite.question for suite in result.suites)
        if result_questions != questions:
            raise ValueError(
                "candidate job suite results do not match request order"
            )


class StubSuiteCompleted(FrozenModel):
    kind: Literal["completed"] = "completed"
    question: MetricQuestionCoordinate
    passed: bool


class StubSuiteHarnessFailure(FrozenModel):
    kind: Literal["harness_failure"] = "harness_failure"
    question: MetricQuestionCoordinate
    failure_type: str
    message: str


StubSuiteResult: TypeAlias = Annotated[
    StubSuiteCompleted | StubSuiteHarnessFailure,
    Field(discriminator="kind"),
]


class StubCandidateJobResult(CandidateJobResult):
    schema_version: Literal[1] = STUB_CANDIDATE_JOB_SCHEMA_VERSION
    candidate: EvalCandidateId
    suites: tuple[StubSuiteResult, ...]


def build_candidate_job_request(
    candidate: MaterializedEvalCandidate,
    suites: tuple[CandidateEvaluatorSuite, ...],
    /,
) -> StubCandidateJobRequest:
    if any(suite.suite_kind != "stub" for suite in suites):
        raise ValueError("stub builder requires stub evaluator suites")
    stub_suites = tuple(
        StubEvaluatorSuite.model_validate(suite.suite_payload)
        for suite in suites
    )
    return StubCandidateJobRequest(candidate=candidate, suites=stub_suites)


def evaluate_stub_candidate_job(request: Jsonable) -> Jsonable:
    validated = StubCandidateJobRequest.model_validate(request)
    namespace: dict[str, object] = {}
    source = validated.candidate.source.source
    try:
        tree = ast.parse(source)
        candidate_code = compile(tree, "<evaluation candidate>", "exec")
        exec(candidate_code, namespace)
    except BaseException:
        return StubCandidateJobResult(
            candidate=validated.candidate.identity,
            suites=(),
        ).model_dump(mode="json", exclude_computed_fields=True)

    suite_results: list[StubSuiteResult] = []
    for suite in validated.suites:
        if suite.support_failure:
            suite_results.append(
                StubSuiteHarnessFailure(
                    question=suite.question,
                    failure_type="RuntimeError",
                    message="support broke",
                )
            )
            continue
        candidate = namespace.get(suite.entry_point)
        if not inspect.isfunction(candidate):
            suite_results.append(
                StubSuiteHarnessFailure(
                    question=suite.question,
                    failure_type="LookupError",
                    message=f"missing entry point {suite.entry_point!r}",
                )
            )
            continue
        passed = True
        for inputs, expected in zip(suite.inputs, suite.expected, strict=True):
            if candidate(*inputs) != expected:
                passed = False
                break
        suite_results.append(
            StubSuiteCompleted(
                question=suite.question,
                passed=passed,
            )
        )

    result = StubCandidateJobResult(
        candidate=validated.candidate.identity,
        suites=tuple(suite_results),
    )
    return result.model_dump(mode="json", exclude_computed_fields=True)
