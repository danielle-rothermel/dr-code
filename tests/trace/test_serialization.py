"""Persistence-boundary tests for complete traces."""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from typing import cast

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


def _attempt_public_mutation(action: Callable[[], object]) -> None:
    """Exercise either a mutable defensive copy or an immutable view."""
    try:
        action()
    except (AttributeError, TypeError):
        pass


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


def test_serialization_is_stable_across_public_mutation_attempts() -> None:
    trace = _full_trace()
    before = serialize_trace(trace).model_dump(mode="json")

    payload_artifact = trace.values["payload"]
    assert isinstance(payload_artifact, JsonArtifact)
    payload = cast(Mapping[str, object], payload_artifact.payload)
    detail = cast(Mapping[str, object], trace.step_facts["parse"]["detail"])
    _attempt_public_mutation(
        lambda: operator.setitem(payload, "task", "mutated")
    )
    _attempt_public_mutation(lambda: operator.setitem(detail, "line", 99))
    _attempt_public_mutation(
        lambda: operator.setitem(
            trace.values,
            "late",
            TextArtifact(text="added through public view"),
        )
    )

    assert serialize_trace(trace).model_dump(mode="json") == before


def test_serialized_trace_rejects_non_finite_step_fact_floats() -> None:
    payload = serialize_trace(_full_trace()).model_dump(mode="python")
    payload["step_facts"]["parse"]["confidence"] = float("inf")

    with pytest.raises(ValidationError):
        SerializedTrace.model_validate(payload)


def test_serialized_trace_rejects_nested_non_finite_json_artifact_float() -> (
    None
):
    payload = serialize_trace(_full_trace()).model_dump(mode="python")
    payload["values"]["payload"]["payload"] = {"nested": [float("nan")]}

    with pytest.raises(ValidationError) as exc_info:
        SerializedTrace.model_validate(payload)

    error = next(
        error
        for error in exc_info.value.errors(include_url=False)
        if error["type"] == "value_error"
    )
    assert error["loc"][:2] == ("values", "payload")
    assert error["loc"][-2:] == ("json", "payload")
    assert str(error["ctx"]["error"]) == (
        "JSON artifact payload must contain only finite floats"
    )
