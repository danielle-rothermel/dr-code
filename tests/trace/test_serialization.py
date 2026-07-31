"""Persistence-boundary tests for complete traces."""

from __future__ import annotations

from dr_code.trace import (
    INPUT_KEY,
    OUTPUT_KEY,
    TRACE_SCHEMA_VERSION,
    Absent,
    CodeArtifact,
    CodeCandidateSetArtifact,
    JsonArtifact,
    SerializedTrace,
    TextArtifact,
    Trace,
    TraceProducer,
    deserialize_trace,
    serialize_trace,
)


def _full_trace() -> Trace:
    return Trace(
        values={
            INPUT_KEY: TextArtifact(text="prompt"),
            OUTPUT_KEY: CodeArtifact(source="x = 1\n"),
            "candidates": CodeCandidateSetArtifact(
                candidates=("a = 1\n", "b = 2\n")
            ),
            "payload": JsonArtifact(payload={"task": "HumanEval/0"}),
            "missing": Absent(
                failed_step="parse",
                cause="syntax error",
                propagated_through=("score",),
            ),
        },
        producer=TraceProducer(
            producer_id="preprocess",
            version="1.0",
            definition_hash="definition-hash",
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


def test_serialized_trace_defaults_schema_version_when_omitted() -> None:
    payload = serialize_trace(_full_trace()).model_dump(mode="json")
    del payload["schema_version"]

    reparsed = SerializedTrace.model_validate(payload)

    assert reparsed.schema_version == TRACE_SCHEMA_VERSION
