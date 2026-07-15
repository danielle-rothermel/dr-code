"""Acceptance tests for serialize/deserialize round-trips."""

from __future__ import annotations

from dr_code.trace import (
    EXTERNAL_PRODUCER,
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

# A trace exercising every artifact kind plus an Absent value.
_FULL_VALUES = {
    INPUT_KEY: TextArtifact(text="prompt"),
    OUTPUT_KEY: CodeArtifact(source="x = 1\n"),
    "candidates": CodeCandidateSetArtifact(candidates=("a", "b")),
    "payload": JsonArtifact(payload={"task": "HumanEval/0"}),
    "missing": Absent(
        failed_step="parse",
        cause="syntax error",
        propagated_through=("score",),
    ),
}


def _full_trace() -> Trace:
    return Trace(
        values=dict(_FULL_VALUES),
        producer=TraceProducer(producer_id="preproc-1", version="1.0"),
        step_facts={"parse": {"reason": "unbalanced parens"}},
    )


# --- serialize -------------------------------------------------------


def test_serialize_stamps_schema_version() -> None:
    serialized = serialize_trace(_full_trace())
    assert serialized.schema_version == TRACE_SCHEMA_VERSION


def test_serialize_carries_producer() -> None:
    trace = _full_trace()
    serialized = serialize_trace(trace)
    assert serialized.producer == trace.producer


def test_serialize_carries_artifacts_and_absences() -> None:
    serialized = serialize_trace(_full_trace())
    # Canonical artifacts AND causal absences are carried (eval-flow L3).
    assert serialized.values[INPUT_KEY] == _FULL_VALUES[INPUT_KEY]
    assert serialized.values["missing"] == _FULL_VALUES["missing"]


def test_serialize_carries_step_facts() -> None:
    serialized = serialize_trace(_full_trace())
    assert serialized.step_facts == {
        "parse": {"reason": "unbalanced parens"}
    }


# --- round-trip through Trace ----------------------------------------


def test_deserialize_restores_values() -> None:
    trace = _full_trace()
    restored = deserialize_trace(serialize_trace(trace))
    for key, value in _FULL_VALUES.items():
        assert restored.value(key) == value


def test_deserialize_restores_producer() -> None:
    trace = _full_trace()
    restored = deserialize_trace(serialize_trace(trace))
    assert restored.producer == trace.producer


def test_deserialize_restores_step_facts() -> None:
    trace = _full_trace()
    restored = deserialize_trace(serialize_trace(trace))
    assert dict(restored.step_facts) == dict(trace.step_facts)


def test_round_trip_is_value_equal_for_external_trace() -> None:
    # Round-trip must be value-equal (S3).
    trace = Trace(
        values={
            INPUT_KEY: TextArtifact(text="in"),
            OUTPUT_KEY: TextArtifact(text="out"),
        },
        producer=EXTERNAL_PRODUCER,
    )
    restored = deserialize_trace(serialize_trace(trace))
    assert restored.value(INPUT_KEY) == trace.value(INPUT_KEY)
    assert restored.value(OUTPUT_KEY) == trace.value(OUTPUT_KEY)
    assert restored.producer == trace.producer


# --- JSON round-trip on SerializedTrace ------------------------------


def test_serialized_trace_json_round_trip_lossless() -> None:
    # SerializedTrace is a BaseModel feeding persistence/external schemas;
    # model_dump_json / re-parse must be lossless, including an Absent
    # value and every artifact kind.
    serialized = serialize_trace(_full_trace())
    reparsed = SerializedTrace.model_validate_json(
        serialized.model_dump_json()
    )
    assert reparsed == serialized


def test_json_round_trip_preserves_absent() -> None:
    serialized = serialize_trace(_full_trace())
    reparsed = SerializedTrace.model_validate_json(
        serialized.model_dump_json()
    )
    assert reparsed.values["missing"] == _FULL_VALUES["missing"]


def test_full_pipeline_json_to_trace_value_equal() -> None:
    trace = _full_trace()
    serialized = serialize_trace(trace)
    reparsed = SerializedTrace.model_validate_json(
        serialized.model_dump_json()
    )
    restored = deserialize_trace(reparsed)
    for key, value in _FULL_VALUES.items():
        assert restored.value(key) == value
    assert restored.producer == trace.producer
