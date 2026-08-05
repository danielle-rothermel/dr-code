"""Persistence-boundary tests for complete traces."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_code.trace import (
    INPUT_KEY,
    OUTPUT_KEY,
    TRACE_SCHEMA_VERSION,
    Absent,
    CandidateInspection,
    CandidateOrigin,
    CodeArtifact,
    CodeCandidate,
    CodeCandidateSetArtifact,
    ComponentCoordinate,
    ExtractionOperation,
    InspectedCodeCandidate,
    InspectedCodeCandidateSetArtifact,
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


def _candidate(source: str, *, location: int) -> CodeCandidate:
    return CodeCandidate(
        source=source,
        origins=(
            CandidateOrigin(
                operation=ExtractionOperation(operation_name="fenced_blocks"),
                input_location=location,
            ),
        ),
    )


def _full_trace() -> Trace:
    return Trace(
        values={
            INPUT_KEY: TextArtifact(text="prompt"),
            OUTPUT_KEY: CodeArtifact(source="x = 1\n"),
            "candidates": CodeCandidateSetArtifact(
                candidates=(
                    _candidate("a = 1\n", location=0),
                    _candidate("b = 2\n", location=1),
                )
            ),
            "inspected": InspectedCodeCandidateSetArtifact(
                candidates=(
                    InspectedCodeCandidate(
                        candidate=_candidate("a = 1\n", location=0),
                        inspection=CandidateInspection(
                            parses=True, compiles=True
                        ),
                    ),
                )
            ),
            "payload": JsonArtifact(payload={"task": "HumanEval/0"}),
            "missing": Absent(
                failed_step="parse",
                failure_code="parse_failed",
                cause="syntax error",
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
        step_facts={
            "parse": {
                "reason": "unbalanced parens",
                "candidate_count": 2,
                "rejected_locations": [1],
                "detail": {"line": 3, "recoverable": False, "hint": None},
                "confidence": 0.5,
            }
        },
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
            "inspected",
            "payload",
            "missing",
        )
    } == {
        TextArtifact,
        CodeArtifact,
        CodeCandidateSetArtifact,
        InspectedCodeCandidateSetArtifact,
        JsonArtifact,
        Absent,
    }


def test_schema_version_is_pinned_to_three() -> None:
    # The persisted schema version is stored identity: pin the literal so a
    # shape change without a version bump fails here.
    assert TRACE_SCHEMA_VERSION == 3
    assert (
        serialize_trace(_full_trace()).model_dump(mode="json")[
            "schema_version"
        ]
        == 3
    )


@pytest.mark.parametrize("schema_version", [None, 1, 2, 4])
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


def test_json_step_facts_survive_the_persistence_round_trip() -> None:
    restored = deserialize_trace(
        SerializedTrace.model_validate_json(
            serialize_trace(_full_trace()).model_dump_json()
        )
    )

    assert restored.step_facts["parse"] == {
        "reason": "unbalanced parens",
        "candidate_count": 2,
        "rejected_locations": [1],
        "detail": {"line": 3, "recoverable": False, "hint": None},
        "confidence": 0.5,
    }


def test_serialized_trace_rejects_non_finite_step_fact_floats() -> None:
    payload = serialize_trace(_full_trace()).model_dump(mode="python")
    payload["step_facts"]["parse"]["confidence"] = float("inf")

    with pytest.raises(ValidationError):
        SerializedTrace.model_validate(payload)
