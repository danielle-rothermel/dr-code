"""Acceptance tests for serialize/deserialize round-trips."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from dr_code.trace import (
    EXTERNAL_PRODUCER_ID,
    INPUT_KEY,
    OUTPUT_KEY,
    TRACE_SCHEMA_VERSION,
    Absent,
    CandidateInspection,
    CandidateLineage,
    CandidateOrigin,
    CodeArtifact,
    CodeCandidateSetArtifact,
    ExtractionOperation,
    ExternalSource,
    IdentifiedCandidate,
    IdentifiedCandidateSetArtifact,
    JsonArtifact,
    SerializedTrace,
    TextArtifact,
    Trace,
    TraceProducer,
    deserialize_trace,
    serialize_trace,
)

_EXTERNAL_PRODUCER = TraceProducer(
    producer_id=EXTERNAL_PRODUCER_ID,
    external_source=ExternalSource(
        source_id="serialization-fixture",
        content_digest="a" * 64,
    ),
)


def _serialized_producer() -> dict[str, str]:
    return {
        "producer_id": "preproc-1",
        "version": "1.0",
        "definition_hash": "d" * 64,
        "preprocessing_config_hash": "c" * 64,
        "implementation_hash": "e" * 64,
    }


def _origin(name: str) -> CandidateOrigin:
    return CandidateOrigin(
        path=(
            ExtractionOperation(
                kind="response_representation", details={"name": name}
            ),
        )
    )


# A trace exercising every artifact kind plus an Absent value.
_FULL_VALUES = {
    INPUT_KEY: TextArtifact(text="prompt"),
    OUTPUT_KEY: CodeArtifact(source="x = 1\n"),
    "candidates": CodeCandidateSetArtifact(
        candidates=("a", "b"),
        lineage=(
            CandidateLineage(
                candidate_id="candidate-a",
                origins=(
                    CandidateOrigin(
                        path=(
                            ExtractionOperation(
                                kind="response_representation",
                                details={"name": "normalized_raw_response"},
                            ),
                            ExtractionOperation(kind="fenced_block"),
                        )
                    ),
                ),
            ),
            CandidateLineage(
                candidate_id="candidate-b",
                origins=(_origin("second"),),
            ),
        ),
    ),
    "payload": JsonArtifact(payload={"task": "HumanEval/0"}),
    "identified": IdentifiedCandidateSetArtifact(
        candidates=(
            IdentifiedCandidate(
                source="def f():\n    return 1",
                lineage=CandidateLineage(
                    candidate_id="candidate-f",
                    origins=(_origin("identified"),),
                ),
                inspection=CandidateInspection(
                    parse_ok=True,
                    parse_error=None,
                    compile_ok=True,
                    compile_error=None,
                    top_level_function_names=("f",),
                ),
            ),
        )
    ),
    "missing": Absent(
        failed_step="parse",
        cause="syntax error",
        failure_code="preprocessing.syntax_error",
        propagated_through=("score",),
    ),
}


def _full_trace() -> Trace:
    return Trace(
        values=dict(_FULL_VALUES),
        producer=TraceProducer(
            producer_id="preproc-1",
            version="1.0",
            definition_hash="d" * 64,
            preprocessing_config_hash="c" * 64,
            implementation_hash="e" * 64,
        ),
        step_facts={
            "parse": {
                "reason": "unbalanced parens",
                "rejected": {"count": 2, "indices": [0, 3]},
            }
        },
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
        "parse": {
            "reason": "unbalanced parens",
            "rejected": {"count": 2, "indices": [0, 3]},
        }
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
        producer=_EXTERNAL_PRODUCER,
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


def test_schema_version_is_required() -> None:
    payload = {
        "producer": _serialized_producer(),
        "values": {
            INPUT_KEY: {"kind": "text", "text": "prompt"},
            OUTPUT_KEY: {"kind": "text", "text": "result"},
        },
    }

    with pytest.raises(ValidationError, match="schema_version"):
        SerializedTrace.model_validate(payload)


@pytest.mark.parametrize("schema_version", [1, 2, 3])
def test_serialized_trace_rejects_other_schema_versions(
    schema_version: int,
) -> None:
    payload = {
        "schema_version": schema_version,
        "producer": _serialized_producer(),
        "values": {
            INPUT_KEY: {"kind": "text", "text": "prompt"},
            OUTPUT_KEY: {"kind": "text", "text": "result"},
        },
    }

    with pytest.raises(ValidationError, match="schema_version"):
        SerializedTrace.model_validate(payload)


def test_schema_v4_absent_requires_failure_code() -> None:
    payload = {
        "schema_version": 4,
        "producer": _serialized_producer(),
        "values": {
            INPUT_KEY: {"kind": "text", "text": "prompt"},
            OUTPUT_KEY: {
                "kind": "absent",
                "failed_step": "parse",
                "cause": "syntax error",
            },
        },
    }

    with pytest.raises(ValidationError, match="failure_code"):
        SerializedTrace.model_validate(payload)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_step_fact_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite float"):
        Trace(
            values={
                INPUT_KEY: TextArtifact(text="in"),
                OUTPUT_KEY: TextArtifact(text="out"),
            },
            producer=_EXTERNAL_PRODUCER,
            step_facts={"parse": {"score": value}},
        )


@pytest.mark.parametrize(
    "facts",
    [
        {"parse": {"value": object()}},
        {"parse": {"value": (1, 2)}},
        {"parse": {1: "non-string key"}},
        {1: {"value": "non-string step name"}},
        {"parse": ["not", "an", "object"]},
    ],
)
def test_non_json_step_fact_is_rejected(facts: object) -> None:
    with pytest.raises(ValueError, match="step_facts"):
        Trace(
            values={
                INPUT_KEY: TextArtifact(text="in"),
                OUTPUT_KEY: TextArtifact(text="out"),
            },
            producer=_EXTERNAL_PRODUCER,
            step_facts=facts,  # type: ignore[arg-type]
        )


def test_mapping_cycle_in_step_facts_is_rejected_with_path() -> None:
    cycle: dict[str, object] = {}
    cycle["self"] = cycle

    with pytest.raises(
        ValueError, match=r"step_facts\.parse\.value\.self.*cycle"
    ):
        Trace(
            values={
                INPUT_KEY: TextArtifact(text="in"),
                OUTPUT_KEY: TextArtifact(text="out"),
            },
            producer=_EXTERNAL_PRODUCER,
            step_facts={"parse": {"value": cycle}},  # type: ignore[dict-item]
        )


def test_list_cycle_in_step_facts_is_rejected_with_path() -> None:
    cycle: list[object] = []
    cycle.append(cycle)

    with pytest.raises(
        ValueError, match=r"step_facts\.parse\.value\[0\].*cycle"
    ):
        Trace(
            values={
                INPUT_KEY: TextArtifact(text="in"),
                OUTPUT_KEY: TextArtifact(text="out"),
            },
            producer=_EXTERNAL_PRODUCER,
            step_facts={"parse": {"value": cycle}},  # type: ignore[dict-item]
        )


def test_shared_acyclic_step_fact_container_is_allowed() -> None:
    shared = {"items": [1, 2]}
    trace = Trace(
        values={
            INPUT_KEY: TextArtifact(text="in"),
            OUTPUT_KEY: TextArtifact(text="out"),
        },
        producer=_EXTERNAL_PRODUCER,
        step_facts={  # type: ignore[dict-item]
            "parse": {"left": shared, "right": shared}
        },
    )

    assert trace.step_facts["parse"] == {
        "left": {"items": [1, 2]},
        "right": {"items": [1, 2]},
    }


def test_full_pipeline_json_to_trace_value_equal() -> None:
    trace = _full_trace()
    serialized = serialize_trace(trace)
    reparsed = SerializedTrace.model_validate_json(
        serialized.model_dump_json()
    )
    restored = deserialize_trace(reparsed)
    assert restored == trace
    for key, value in _FULL_VALUES.items():
        assert restored.value(key) == value
    assert restored.producer == trace.producer


def test_deserialize_trace_rejects_bypassed_schema_validation() -> None:
    serialized = serialize_trace(_full_trace()).model_copy(
        update={"schema_version": 1}
    )

    with pytest.raises(ValueError, match="unsupported.*schema version"):
        deserialize_trace(serialized)
