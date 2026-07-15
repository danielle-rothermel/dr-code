"""Serialized-trace contract for traces from named definitions.

``serialize_trace`` / ``deserialize_trace`` must round-trip a preprocessing
``Trace`` losslessly: every artifact value, every ``Absent`` with its causal
lineage, every step fact, the producer id/version, and the artifact kinds all
survive — including through a JSON model round-trip (the persistence path).
Traces are produced by the real named definitions, so the producer identity
under test is the one the resolver stamps.
"""

from __future__ import annotations

from dr_code.preprocessing import (
    resolve_preprocessing_definition,
    run_preprocessing,
)
from dr_code.preprocessing.definition import preprocessing_definition_hash
from dr_code.trace import (
    Absent,
    CodeArtifact,
    CodeCandidateSetArtifact,
    SerializedTrace,
    TextArtifact,
    Trace,
    deserialize_trace,
    is_absent,
    serialize_trace,
)

BEST_EFFORT_ID = "humaneval-best-effort"
FIELD_MARKER_ID = "humaneval-field-marker"

_FENCED = "Here is the code:\n```python\ndef f(x):\n    return x + 1\n```\n"


def _best_effort_v2():
    return resolve_preprocessing_definition(
        definition_id=BEST_EFFORT_ID, version="v2"
    )


def _trace(raw: str) -> Trace:
    return run_preprocessing(_best_effort_v2(), TextArtifact(text=raw))


def _assert_round_trip(trace: Trace) -> Trace:
    serialized = serialize_trace(trace)
    assert isinstance(serialized, SerializedTrace)
    restored = deserialize_trace(serialized)
    # Values and step facts survive verbatim.
    assert dict(restored.values) == dict(trace.values)
    assert dict(restored.step_facts) == dict(trace.step_facts)
    assert restored.producer == trace.producer
    return restored


def _assert_json_round_trip(trace: Trace) -> Trace:
    # The persistence path: SerializedTrace is a pydantic model, so it must
    # survive a model_dump(mode="json") -> model_validate round-trip too.
    serialized = serialize_trace(trace)
    payload = serialized.model_dump(mode="json")
    reserialized = SerializedTrace.model_validate(payload)
    assert reserialized == serialized
    restored = deserialize_trace(reserialized)
    assert dict(restored.values) == dict(trace.values)
    return restored


# --- success trace: values, kinds, facts, producer survive -----------


def test_round_trip_preserves_code_output_and_facts() -> None:
    trace = _trace(_FENCED)
    assert isinstance(trace.value("output"), CodeArtifact)

    restored = _assert_round_trip(trace)
    out = restored.value("output")
    assert isinstance(out, CodeArtifact)
    assert out.source == trace.value("output").source
    # The extraction alternative fact is preserved.
    assert restored.step_facts["extract_candidates"] == {
        "alternative": "fenced_blocks"
    }


def test_round_trip_preserves_producer_id_and_version() -> None:
    trace = _trace(_FENCED)
    restored = _assert_round_trip(trace)
    assert restored.producer.producer_id == BEST_EFFORT_ID
    assert restored.producer.version == "v2"
    assert restored.producer.definition_hash == preprocessing_definition_hash(
        _best_effort_v2()
    )


def test_round_trip_preserves_input_artifact_kind() -> None:
    trace = _trace(_FENCED)
    restored = _assert_round_trip(trace)
    restored_input = restored.value("input")
    assert isinstance(restored_input, TextArtifact)
    assert restored_input == TextArtifact(text=_FENCED)


def test_round_trip_preserves_candidate_set_kind() -> None:
    # An intermediate candidate-set value keeps its concrete artifact kind.
    trace = _trace(_FENCED)
    restored = _assert_round_trip(trace)
    extracted = restored.value("extract_candidates")
    assert isinstance(extracted, CodeCandidateSetArtifact)
    assert extracted == trace.value("extract_candidates")


def test_json_round_trip_is_lossless() -> None:
    _assert_json_round_trip(_trace(_FENCED))


# --- absent trace: causal lineage and propagation survive ------------


def test_round_trip_preserves_absent_output_and_lineage() -> None:
    # Prose-only input recovers no candidate: the output is Absent and its
    # failed_step / propagated_through lineage must survive serialization.
    trace = _trace("Just an explanation, no code at all.\n")
    output = trace.value("output")
    assert is_absent(output)

    restored = _assert_round_trip(trace)
    restored_output = restored.value("output")
    assert isinstance(restored_output, Absent)
    assert restored_output.failed_step == output.failed_step
    assert restored_output.cause == output.cause
    assert restored_output.propagated_through == output.propagated_through


def test_round_trip_preserves_absent_propagation_through_steps() -> None:
    # After the failing step, every downstream value is the same Absent with
    # propagated_through extended — that propagation must survive too.
    trace = _trace("Just an explanation, no code at all.\n")
    restored = _assert_round_trip(trace)
    absent_values = [
        (key, value)
        for key, value in restored.values.items()
        for _ in [0]
        if is_absent(value)
    ]
    assert absent_values, "expected at least one Absent value in the trace"
    for key, value in absent_values:
        original = trace.value(key)
        assert isinstance(original, Absent)
        assert value.failed_step == original.failed_step
        assert value.propagated_through == original.propagated_through


def test_json_round_trip_is_lossless_for_absent_trace() -> None:
    _assert_json_round_trip(_trace("no code here\n"))
