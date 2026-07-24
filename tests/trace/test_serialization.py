"""Persistence-boundary tests for complete traces."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_code.trace import (
    INPUT_KEY,
    OUTPUT_KEY,
    TRACE_SCHEMA_VERSION,
    Absent,
    CandidateLineage,
    CandidateOrigin,
    CodeArtifact,
    CodeCandidateSetArtifact,
    ComponentCoordinate,
    ExtractionOperation,
    JsonArtifact,
    PreprocessingDefinitionCoordinate,
    PreprocessingTraceProducer,
    SerializedTrace,
    StepCoordinate,
    TextArtifact,
    Trace,
    deserialize_trace,
    serialize_trace,
)


def _lineage(name: str) -> CandidateLineage:
    """A minimal well-formed lineage for one candidate."""
    return CandidateLineage(
        origins=(
            CandidateOrigin(
                path=(
                    ExtractionOperation(
                        kind="response_representation",
                        details={"name": name},
                    ),
                )
            ),
        )
    )


def _full_trace() -> Trace:
    return Trace(
        values={
            INPUT_KEY: TextArtifact(text="prompt"),
            OUTPUT_KEY: CodeArtifact(source="x = 1\n"),
            "candidates": CodeCandidateSetArtifact(
                candidates=("a = 1\n", "b = 2\n"),
                lineage=(_lineage("first"), _lineage("second")),
            ),
            "payload": JsonArtifact(payload={"task": "HumanEval/0"}),
            "missing": Absent(
                failed_step="parse",
                cause="syntax error",
                failure_code="test.syntax_error",
                propagated_through=("score",),
            ),
        },
        producer=PreprocessingTraceProducer(
            definition=PreprocessingDefinitionCoordinate(
                definition_id="preprocess",
                version="1.0",
                steps=(
                    StepCoordinate(
                        instance_name="parse",
                        component=ComponentCoordinate(
                            registered_name="normalize_unicode",
                            version="0",
                        ),
                    ),
                ),
            )
        ),
        step_facts={"parse": {"reason": "unbalanced parens"}},
    )


def test_json_round_trip_preserves_full_trace_union() -> None:
    trace = _full_trace()
    serialized = serialize_trace(trace)
    reparsed = SerializedTrace.model_validate_json(
        serialized.model_dump_json()
    )
    restored = deserialize_trace(reparsed)

    assert serialized.schema_version == TRACE_SCHEMA_VERSION
    assert restored == trace
    assert {
        type(restored.value(key))
        for key in (
            INPUT_KEY,
            OUTPUT_KEY,
            "candidates",
            "payload",
            "missing",
        )
    } == {
        TextArtifact,
        CodeArtifact,
        CodeCandidateSetArtifact,
        JsonArtifact,
        Absent,
    }


@pytest.mark.parametrize("schema_version", [None, 1, 3])
def test_serialized_trace_rejects_missing_or_unsupported_schema_version(
    schema_version: int | None,
) -> None:
    payload = serialize_trace(_full_trace()).model_dump(mode="json")
    if schema_version is None:
        del payload["schema_version"]
    else:
        payload["schema_version"] = schema_version

    with pytest.raises(ValidationError):
        SerializedTrace.model_validate(payload)
