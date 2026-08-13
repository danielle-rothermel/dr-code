from __future__ import annotations

import math
from typing import cast

from dr_exec import (
    BudgetExceededOutcome,
    Budgets,
    CompletedExecution,
    ExecutionJob,
    ExitedOutcome,
    FiniteDurationLimit,
    JobId,
    ProtocolFailedOutcome,
    build_in_process_importable_json_job,
    parse_importable_json_result,
)
from dr_serialize import Jsonable

from dr_code.preprocessing.definition import PreprocessingDefinition
from dr_code.preprocessing.job import (
    CANDIDATE_SOURCES_ENTRY_POINT,
    PREPROCESS_TEXT_ENTRY_POINT,
    CandidateSourcesJobResult,
    PreprocessTextJobRequest,
    PreprocessTextJobResult,
)
from dr_code.trace import Trace, deserialize_trace


_NANOSECONDS_PER_SECOND = 1_000_000_000


class PreprocessTextExecutionError(Exception):
    """A preprocessing job did not return one valid serialized trace."""


class PreprocessTextTimeoutError(PreprocessTextExecutionError):
    """A preprocessing job exceeded its declared wall-time budget.

    Raised only when the worker pool killed the job on the wall-time axis, so
    a wedged input is distinguishable from every other per-item failure.
    """


def preprocess_job_budgets(wall_time_seconds: float, /) -> Budgets:
    """Return per-job budgets carrying one finite wall-time limit.

    Only the wall-time axis is bounded: preprocessing is trusted, in-process
    work whose sole failure mode worth guarding is a job that never returns.
    """

    if wall_time_seconds <= 0 or not math.isfinite(wall_time_seconds):
        raise ValueError("wall_time_seconds must be finite and positive")
    wall_time_nanoseconds = wall_time_seconds * _NANOSECONDS_PER_SECOND
    if not math.isfinite(wall_time_nanoseconds):
        raise ValueError("wall_time_seconds is too large to represent")
    return Budgets(
        wall_time=FiniteDurationLimit(max_ns=math.ceil(wall_time_nanoseconds))
    )


def preprocess_text_request(
    text: str,
    definition: PreprocessingDefinition,
    /,
) -> PreprocessTextJobRequest:
    return PreprocessTextJobRequest(
        text=text,
        definition_id=definition.definition_id,
        definition_version=definition.version,
    )


def build_preprocess_text_job(
    job_id: JobId,
    request: PreprocessTextJobRequest,
    /,
    *,
    budgets: Budgets | None = None,
) -> ExecutionJob:
    return build_in_process_importable_json_job(
        job_id,
        PREPROCESS_TEXT_ENTRY_POINT,
        _request_payload(request),
        budgets=budgets,
    )


def build_candidate_sources_job(
    job_id: JobId,
    request: PreprocessTextJobRequest,
    /,
    *,
    budgets: Budgets | None = None,
) -> ExecutionJob:
    return build_in_process_importable_json_job(
        job_id,
        CANDIDATE_SOURCES_ENTRY_POINT,
        _request_payload(request),
        budgets=budgets,
    )


def parse_preprocess_text_result(completed: CompletedExecution, /) -> Trace:
    payload = _clean_result_payload(completed)
    try:
        result = PreprocessTextJobResult.model_validate(payload)
    except Exception as error:
        raise PreprocessTextExecutionError(str(error)) from error
    return deserialize_trace(result.trace)


def parse_candidate_sources_result(
    completed: CompletedExecution,
    /,
) -> tuple[str, ...]:
    payload = _clean_result_payload(completed)
    try:
        result = CandidateSourcesJobResult.model_validate(payload)
    except Exception as error:
        raise PreprocessTextExecutionError(str(error)) from error
    return result.sources


def _clean_result_payload(completed: CompletedExecution, /) -> Jsonable:
    outcome = completed.result.outcome
    if isinstance(outcome, BudgetExceededOutcome):
        raise PreprocessTextTimeoutError(
            f"preprocessing job exceeded its {outcome.axis} budget"
        )
    if isinstance(outcome, ProtocolFailedOutcome):
        raise PreprocessTextExecutionError(outcome.failure_detail or outcome)
    if not isinstance(outcome, ExitedOutcome) or outcome.exit_code != 0:
        raise PreprocessTextExecutionError(
            "preprocessing job did not exit cleanly with code zero"
        )
    try:
        return parse_importable_json_result(completed)
    except Exception as error:
        raise PreprocessTextExecutionError(str(error)) from error


def _request_payload(request: PreprocessTextJobRequest) -> Jsonable:
    return cast(
        Jsonable,
        request.model_dump(mode="json", exclude_computed_fields=True),
    )


__all__ = [
    "PreprocessTextExecutionError",
    "PreprocessTextTimeoutError",
    "build_candidate_sources_job",
    "build_preprocess_text_job",
    "parse_candidate_sources_result",
    "parse_preprocess_text_result",
    "preprocess_job_budgets",
    "preprocess_text_request",
]
