from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib.metadata import entry_points
from types import MappingProxyType
from typing import Final, Protocol

from dr_exec import ImportableEntryPoint
from dr_serialize import IdentityDocument, Jsonable, build_identity_document
from pydantic import Field

from dr_code.core.models import FrozenModel
from dr_code.evaluation.id import EvalCandidateId, MaterializedEvalCandidate
from dr_code.metrics.coordinates import MetricQuestionCoordinate

_CANDIDATE_JOB_BUILDER_GROUP: Final = "dr_code.candidate_job_builders"


class CandidateJobResult(FrozenModel):
    schema_version: int
    candidate: EvalCandidateId


class CandidateEvaluatorSuite(FrozenModel):
    question: MetricQuestionCoordinate
    suite_kind: str = Field(min_length=1)
    suite_payload: Jsonable


class CandidateJobRequest(Protocol):
    candidate: MaterializedEvalCandidate

    def execution_entry_point(self) -> ImportableEntryPoint: ...

    def request_identity_schema(self) -> str: ...

    def request_identity_schema_version(self) -> int: ...

    def request_payload(self) -> Jsonable: ...

    def result_model(self) -> type[CandidateJobResult]: ...

    def validate_result_matches_request(
        self,
        result: CandidateJobResult,
        /,
    ) -> None: ...

    def model_dump(
        self,
        *,
        mode: str = "python",
        exclude_computed_fields: bool = False,
    ) -> dict[str, object]: ...


CandidateJobBuilder = Callable[
    [MaterializedEvalCandidate, tuple[CandidateEvaluatorSuite, ...]],
    CandidateJobRequest,
]


def _load_candidate_job_builders() -> Mapping[str, CandidateJobBuilder]:
    builders: dict[str, CandidateJobBuilder] = {}
    for entry_point in entry_points(group=_CANDIDATE_JOB_BUILDER_GROUP):
        builder = entry_point.load()
        builders[entry_point.name] = builder
    return builders


_CANDIDATE_JOB_BUILDERS: Mapping[str, CandidateJobBuilder] | None = None


def _candidate_job_builders() -> Mapping[str, CandidateJobBuilder]:
    global _CANDIDATE_JOB_BUILDERS
    if _CANDIDATE_JOB_BUILDERS is None:
        _CANDIDATE_JOB_BUILDERS = MappingProxyType(
            _load_candidate_job_builders()
        )
    return _CANDIDATE_JOB_BUILDERS


def register_candidate_job_builder(
    name: str,
    builder: CandidateJobBuilder,
    /,
) -> None:
    global _CANDIDATE_JOB_BUILDERS
    updated = dict(_candidate_job_builders())
    updated[name] = builder
    _CANDIDATE_JOB_BUILDERS = MappingProxyType(updated)


def build_candidate_job_request(
    candidate: MaterializedEvalCandidate,
    suites: tuple[CandidateEvaluatorSuite, ...],
    /,
) -> CandidateJobRequest:
    if not suites:
        raise ValueError("at least one evaluator suite is required")
    suite_kind = suites[0].suite_kind
    if any(suite.suite_kind != suite_kind for suite in suites):
        raise ValueError("evaluator suites must share one suite_kind")
    try:
        builder = _candidate_job_builders()[suite_kind]
    except KeyError as error:
        raise ValueError(
            f"no candidate job builder registered for suite_kind {suite_kind!r}"
        ) from error
    return builder(candidate, suites)


def candidate_request_identity_document(
    request: CandidateJobRequest,
    /,
) -> IdentityDocument:
    return build_identity_document(
        schema=request.request_identity_schema(),
        schema_version=request.request_identity_schema_version(),
        payload=request.request_payload(),
    )


__all__ = [
    "CandidateEvaluatorSuite",
    "CandidateJobRequest",
    "CandidateJobResult",
    "build_candidate_job_request",
    "candidate_request_identity_document",
    "register_candidate_job_builder",
]
