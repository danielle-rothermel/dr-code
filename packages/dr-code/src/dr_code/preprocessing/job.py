from __future__ import annotations

from typing import Final, Literal

from dr_exec import ImportableEntryPoint
from dr_serialize import Jsonable
from pydantic import StrictStr

from dr_code.core.models import FrozenModel
from dr_code.preprocessing import (
    bind_preprocessing,
    resolve_preprocessing_definition,
)
from dr_code.trace import (
    OUTPUT_KEY,
    Absent,
    InspectedCodeCandidateSetArtifact,
    TextArtifact,
    serialize_trace,
)
from dr_code.trace.serialization import SerializedTrace

PREPROCESS_TEXT_JOB_SCHEMA_VERSION: Final = 1
PREPROCESS_TEXT_ENTRY_POINT: Final = ImportableEntryPoint(
    module_name="dr_code.preprocessing.job",
    attribute_name="preprocess_text_job",
)
CANDIDATE_SOURCES_ENTRY_POINT: Final = ImportableEntryPoint(
    module_name="dr_code.preprocessing.job",
    attribute_name="candidate_sources_job",
)


class PreprocessTextJobRequest(FrozenModel):
    schema_version: Literal[1] = PREPROCESS_TEXT_JOB_SCHEMA_VERSION
    text: StrictStr
    definition_id: StrictStr
    definition_version: StrictStr


class PreprocessTextJobResult(FrozenModel):
    schema_version: Literal[1] = PREPROCESS_TEXT_JOB_SCHEMA_VERSION
    trace: SerializedTrace


class CandidateSourcesJobResult(FrozenModel):
    """The candidate sources one text yielded, without its trace."""

    schema_version: Literal[1] = PREPROCESS_TEXT_JOB_SCHEMA_VERSION
    sources: tuple[StrictStr, ...]


def preprocess_text_job(request: Jsonable) -> Jsonable:
    """Run one trusted preprocessing definition over one text input."""

    validated = PreprocessTextJobRequest.model_validate(request)
    definition = resolve_preprocessing_definition(
        definition_id=validated.definition_id,
        version=validated.definition_version,
    )
    runner = bind_preprocessing(definition)
    trace = runner.run(TextArtifact(text=validated.text))
    result = PreprocessTextJobResult(trace=serialize_trace(trace))
    return result.model_dump(mode="json", exclude_computed_fields=True)


def candidate_sources_job(request: Jsonable) -> Jsonable:
    """Run one preprocessing definition and return only candidate sources.

    Callers that consume candidate sources rather than whole traces use this
    entry point so the trace stays in the worker: a serialized trace is two
    orders of magnitude larger than the sources it carries, and every byte
    crossing the boundary is decoded and validated in the caller.
    """

    validated = PreprocessTextJobRequest.model_validate(request)
    definition = resolve_preprocessing_definition(
        definition_id=validated.definition_id,
        version=validated.definition_version,
    )
    runner = bind_preprocessing(definition)
    trace = runner.run(TextArtifact(text=validated.text))
    result = CandidateSourcesJobResult(
        sources=_candidate_sources(trace.value(OUTPUT_KEY))
    )
    return result.model_dump(mode="json", exclude_computed_fields=True)


def _candidate_sources(trace_output: object) -> tuple[str, ...]:
    if isinstance(trace_output, Absent):
        return ()
    if not isinstance(trace_output, InspectedCodeCandidateSetArtifact):
        raise TypeError("preprocessing did not return inspected candidates")
    sources: list[str] = []
    for inspected in trace_output.candidates:
        if not inspected.inspection.compiles:
            raise RuntimeError("final candidate does not compile")
        if not inspected.inspection.top_level_function_names:
            raise RuntimeError("final candidate has no top-level function")
        sources.append(inspected.candidate.source)
    return tuple(sources)


__all__ = [
    "CANDIDATE_SOURCES_ENTRY_POINT",
    "PREPROCESS_TEXT_ENTRY_POINT",
    "PREPROCESS_TEXT_JOB_SCHEMA_VERSION",
    "CandidateSourcesJobResult",
    "PreprocessTextJobRequest",
    "PreprocessTextJobResult",
    "candidate_sources_job",
    "preprocess_text_job",
]
