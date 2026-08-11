from __future__ import annotations

from typing import cast

from dr_exec import (
    CompletedExecution,
    ExecutionJob,
    ExitedOutcome,
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


class PreprocessTextExecutionError(Exception):
    """A preprocessing job did not return one valid serialized trace."""


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
) -> ExecutionJob:
    return build_in_process_importable_json_job(
        job_id,
        PREPROCESS_TEXT_ENTRY_POINT,
        _request_payload(request),
    )


def build_candidate_sources_job(
    job_id: JobId,
    request: PreprocessTextJobRequest,
    /,
) -> ExecutionJob:
    return build_in_process_importable_json_job(
        job_id,
        CANDIDATE_SOURCES_ENTRY_POINT,
        _request_payload(request),
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
    "build_candidate_sources_job",
    "build_preprocess_text_job",
    "parse_candidate_sources_result",
    "parse_preprocess_text_result",
    "preprocess_text_request",
]
