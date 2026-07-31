"""Acceptance tests for Trace, reserved keys, and WiringError."""

from __future__ import annotations

import pytest

from dr_code.trace import (
    EXTERNAL_PRODUCER_ID,
    INPUT_KEY,
    OUTPUT_KEY,
    RESERVED_KEYS,
    Absent,
    ExternalSource,
    JsonArtifact,
    TextArtifact,
    Trace,
    TraceProducer,
    WiringError,
    external_trace,
)

_SOURCE = ExternalSource(source_id="trace-fixture", content_digest="a" * 64)


def _producer():
    from dr_code.trace import TraceProducer

    return TraceProducer(
        producer_id=EXTERNAL_PRODUCER_ID,
        external_source=_SOURCE,
    )


def _minimal_values() -> dict[str, object]:
    # A trace must contain both reserved keys.
    return {
        INPUT_KEY: TextArtifact(text="in"),
        OUTPUT_KEY: TextArtifact(text="out"),
    }


# --- reserved keys ---------------------------------------------------


def test_reserved_keys_are_input_and_output() -> None:
    assert INPUT_KEY == "input"
    assert OUTPUT_KEY == "output"
    assert RESERVED_KEYS == frozenset({"input", "output"})


def test_wiring_error_is_exception_subclass() -> None:
    assert issubclass(WiringError, Exception)


# --- construction / __post_init__ ------------------------------------


def test_trace_constructs_with_reserved_keys() -> None:
    trace = Trace(values=_minimal_values(), producer=_producer())
    assert trace.producer == _producer()


def test_trace_missing_reserved_key_rejected() -> None:
    # __post_init__ validates RESERVED_KEYS subset of values.
    with pytest.raises((WiringError, ValueError)):
        Trace(
            values={INPUT_KEY: TextArtifact(text="input")},
            producer=_producer(),
        )


def test_trace_rejects_non_trace_values() -> None:
    values: dict[str, object] = _minimal_values()
    values["invalid"] = "not a trace value"

    with pytest.raises(
        WiringError,
        match="value for key 'invalid' is not a TraceValue: str",
    ):
        Trace(values=values, producer=_producer())  # type: ignore[arg-type]


# --- value() lookup --------------------------------------------------


def test_value_returns_present_artifact() -> None:
    art = TextArtifact(text="in")
    trace = Trace(
        values={INPUT_KEY: art, OUTPUT_KEY: TextArtifact(text="out")},
        producer=_producer(),
    )
    assert trace.value(INPUT_KEY) == art


def test_value_missing_key_raises_wiring_error() -> None:
    # Missing key raises WiringError (incompatible definitions).
    trace = Trace(values=_minimal_values(), producer=_producer())
    with pytest.raises(WiringError):
        trace.value("not_present")


def test_value_present_but_absent_returns_absent() -> None:
    # Present-but-Absent is data: value() returns the Absent, not raise.
    absent = Absent(
        failed_step="parse",
        cause="boom",
        failure_code="test.parse_failure",
    )
    trace = Trace(
        values={
            INPUT_KEY: TextArtifact(text="in"),
            OUTPUT_KEY: absent,
        },
        producer=_producer(),
    )
    assert trace.value(OUTPUT_KEY) == absent


# --- external_trace --------------------------------------------------


def test_external_trace_stamps_external_producer() -> None:
    trace = external_trace(_minimal_values(), source=_SOURCE)
    assert trace.producer == _producer()


def test_external_trace_carries_values() -> None:
    art = TextArtifact(text="in")
    trace = external_trace(
        {INPUT_KEY: art, OUTPUT_KEY: TextArtifact(text="out")},
        source=_SOURCE,
    )
    assert trace.value(INPUT_KEY) == art


def test_external_trace_validates_value_types() -> None:
    # external_trace validates value types on the way in.
    with pytest.raises((WiringError, ValueError, TypeError)):
        external_trace(
            {
                INPUT_KEY: TextArtifact(text="in"),
                OUTPUT_KEY: "not a trace value",  # type: ignore[dict-item]
            },
            source=_SOURCE,
        )


def test_external_trace_accepts_step_facts() -> None:
    trace = external_trace(
        _minimal_values(),
        source=_SOURCE,
        step_facts={
            "step_a": {
                "chosen": "candidate_0",
                "candidates": {"accepted": [0, 2], "rejected": [1]},
            }
        },
    )
    assert trace.step_facts["step_a"] == {
        "chosen": "candidate_0",
        "candidates": {"accepted": [0, 2], "rejected": [1]},
    }


def test_trace_snapshots_and_defensively_copies_nested_boundary_data() -> None:
    payload = {"nested": {"items": [1]}}
    facts = {"step": {"nested": {"items": [2]}}}
    artifact = JsonArtifact(payload=payload)
    values = {
        INPUT_KEY: artifact,
        OUTPUT_KEY: TextArtifact(text="out"),
    }
    trace = external_trace(values, source=_SOURCE, step_facts=facts)

    payload["nested"]["items"].append(3)
    artifact.payload["nested"]["items"].append(8)
    facts["step"]["nested"]["items"].append(4)
    values[OUTPUT_KEY] = TextArtifact(text="changed")

    input_value = trace.value(INPUT_KEY)
    assert isinstance(input_value, JsonArtifact)
    assert input_value.payload == {"nested": {"items": [1]}}
    assert trace.step_facts["step"] == {"nested": {"items": [2]}}
    assert trace.value(OUTPUT_KEY) == TextArtifact(text="out")

    input_value.payload["nested"]["items"].append(5)
    exposed_values = trace.values
    exposed_input = exposed_values[INPUT_KEY]
    assert isinstance(exposed_input, JsonArtifact)
    exposed_input.payload["nested"]["items"].append(6)
    exposed_facts = trace.step_facts
    exposed_facts["step"]["nested"]["items"].append(7)

    assert trace.value(INPUT_KEY).payload == {"nested": {"items": [1]}}
    assert trace.step_facts["step"] == {"nested": {"items": [2]}}


# --- producer identity -----------------------------------------------


def test_trace_records_non_external_producer() -> None:
    producer = TraceProducer(
        producer_id="preproc-1",
        version="1.0",
        definition_hash="d" * 64,
        preprocessing_config_hash="c" * 64,
        implementation_hash="e" * 64,
    )
    trace = Trace(values=_minimal_values(), producer=producer)
    assert trace.producer.producer_id == "preproc-1"
    assert trace.producer.preprocessing_config_hash == "c" * 64
