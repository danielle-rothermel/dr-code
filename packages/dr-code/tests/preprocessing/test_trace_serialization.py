from __future__ import annotations

from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID,
    PreprocessingFailureCode,
    bind_preprocessing,
)
from dr_code.trace import (
    Absent,
    CodeCandidateSetArtifact,
    InspectedCodeCandidateSetArtifact,
    SerializedTrace,
    TextArtifact,
    Trace,
    deserialize_trace,
    is_absent,
    serialize_trace,
)

_FENCED = "Here is the code:\n```python\ndef f(x):\n    return x + 1\n```\n"

_RUNNER = bind_preprocessing(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION)


def _trace(raw: str) -> Trace:
    return _RUNNER.run(TextArtifact(text=raw))


def _assert_round_trip(trace: Trace) -> Trace:
    serialized = serialize_trace(trace)
    assert isinstance(serialized, SerializedTrace)
    restored = deserialize_trace(serialized)

    assert dict(restored.values) == dict(trace.values)
    assert dict(restored.step_facts) == dict(trace.step_facts)
    assert restored.producer == trace.producer
    return restored


def _assert_json_round_trip(trace: Trace) -> Trace:
    serialized = serialize_trace(trace)
    payload = serialized.model_dump(mode="json")
    reserialized = SerializedTrace.model_validate(payload)
    assert reserialized == serialized
    restored = deserialize_trace(reserialized)
    assert dict(restored.values) == dict(trace.values)
    return restored


def test_round_trip_preserves_inspected_output_and_facts() -> None:
    trace = _trace(_FENCED)
    original = trace.value("output")
    assert isinstance(original, InspectedCodeCandidateSetArtifact)

    restored = _assert_round_trip(trace)
    out = restored.value("output")
    assert isinstance(out, InspectedCodeCandidateSetArtifact)
    assert out == original

    assert all(item.inspection.compiles for item in out.candidates)

    assert (
        restored.step_facts["extract_all_representations"]
        == (trace.step_facts["extract_all_representations"])
    )


def test_round_trip_preserves_structured_producer_coordinate() -> None:
    trace = _trace(_FENCED)
    restored = _assert_round_trip(trace)
    assert restored.producer.kind == "preprocessing"
    assert restored.producer.definition.definition_id == (
        EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION_ID
    )
    assert restored.producer.definition.version == "0"
    assert restored.producer.definition.steps
    assert {
        step.component.version for step in restored.producer.definition.steps
    } == {"0"}


def test_round_trip_preserves_input_artifact_kind() -> None:
    trace = _trace(_FENCED)
    restored = _assert_round_trip(trace)
    restored_input = restored.value("input")
    assert isinstance(restored_input, TextArtifact)
    assert restored_input == TextArtifact(text=_FENCED)


def test_round_trip_preserves_candidate_set_kind() -> None:
    trace = _trace(_FENCED)
    restored = _assert_round_trip(trace)
    extracted = restored.value("extract_all_representations")
    assert isinstance(extracted, CodeCandidateSetArtifact)
    assert extracted == trace.value("extract_all_representations")


def test_round_trip_preserves_candidate_lineage() -> None:
    trace = _trace(_FENCED)
    restored = _assert_json_round_trip(trace)
    cleaned = restored.value("dedupe_imports")
    assert isinstance(cleaned, CodeCandidateSetArtifact)

    for candidate in cleaned.candidates:
        operations = [
            origin.operation.operation_name for origin in candidate.origins
        ]
        assert operations[1:] == [
            "strip_fences",
            "dedent_candidates",
            "normalize_text_preserving_semantics",
            "normalize_smart_quotes",
            "split_on_name_guard",
            "repair_import_lines",
            "infer_missing_imports",
            "dedupe_imports",
        ]


def test_round_trip_preserves_merged_dedupe_lineage() -> None:
    trace = _trace(_FENCED)
    restored = _assert_json_round_trip(trace)
    merged = restored.value("dedupe_candidates")
    assert isinstance(merged, CodeCandidateSetArtifact)
    assert merged == trace.value("dedupe_candidates")


def test_json_round_trip_is_lossless() -> None:
    _assert_json_round_trip(_trace(_FENCED))


def test_round_trip_preserves_absent_output_and_lineage() -> None:
    trace = _trace("Just an explanation, no code at all.\n")
    output = trace.value("output")
    assert is_absent(output)

    restored = _assert_round_trip(trace)
    restored_output = restored.value("output")
    assert isinstance(restored_output, Absent)
    assert restored_output.failed_step == output.failed_step
    assert restored_output.failure_code == output.failure_code
    assert restored_output.cause == output.cause
    assert restored_output.propagated_through == output.propagated_through


def test_round_trip_preserves_the_producer_failure_code() -> None:
    trace = _trace("Just an explanation, no code at all.\n")
    output = trace.value("output")
    assert is_absent(output)
    assert output.failure_code == (
        PreprocessingFailureCode.NO_CANDIDATE_SURVIVED_FILTERING
    )

    restored = _assert_json_round_trip(trace)
    restored_output = restored.value("output")
    assert isinstance(restored_output, Absent)
    assert restored_output.failure_code == (
        PreprocessingFailureCode.NO_CANDIDATE_SURVIVED_FILTERING
    )


def test_round_trip_preserves_absent_propagation_through_steps() -> None:
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
