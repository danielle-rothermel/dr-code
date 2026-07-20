"""Acceptance tests for Trace, reserved keys, and WiringError."""

from __future__ import annotations

import pytest

from dr_code.trace import (
    EXTERNAL_PRODUCER,
    INPUT_KEY,
    OUTPUT_KEY,
    RESERVED_KEYS,
    Absent,
    TextArtifact,
    Trace,
    TraceProducer,
    WiringError,
    external_trace,
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
    trace = Trace(
        values=_minimal_values(), producer=EXTERNAL_PRODUCER
    )
    assert trace.producer == EXTERNAL_PRODUCER


def test_trace_missing_reserved_key_rejected() -> None:
    # __post_init__ validates RESERVED_KEYS subset of values.
    with pytest.raises((WiringError, ValueError)):
        Trace(
            values={INPUT_KEY: TextArtifact(text="in")},
            producer=EXTERNAL_PRODUCER,
        )


def test_trace_rejects_non_trace_value_entry() -> None:
    # __post_init__ rejects non-TraceValue entries.
    bad = _minimal_values()
    bad["extra"] = "not a trace value"
    with pytest.raises((WiringError, ValueError, TypeError)):
        Trace(values=bad, producer=EXTERNAL_PRODUCER)


# --- value() lookup --------------------------------------------------


def test_value_returns_present_artifact() -> None:
    art = TextArtifact(text="in")
    trace = Trace(
        values={INPUT_KEY: art, OUTPUT_KEY: TextArtifact(text="out")},
        producer=EXTERNAL_PRODUCER,
    )
    assert trace.value(INPUT_KEY) == art


def test_value_missing_key_raises_wiring_error() -> None:
    # Missing key raises WiringError (incompatible definitions).
    trace = Trace(values=_minimal_values(), producer=EXTERNAL_PRODUCER)
    with pytest.raises(WiringError):
        trace.value("not_present")


def test_value_present_but_absent_returns_absent() -> None:
    # Present-but-Absent is data: value() returns the Absent, not raise.
    absent = Absent(failed_step="parse", cause="boom")
    trace = Trace(
        values={
            INPUT_KEY: TextArtifact(text="in"),
            OUTPUT_KEY: absent,
        },
        producer=EXTERNAL_PRODUCER,
    )
    assert trace.value(OUTPUT_KEY) == absent


# --- external_trace --------------------------------------------------


def test_external_trace_stamps_external_producer() -> None:
    # external_trace stamps producer=EXTERNAL_PRODUCER (X-S2).
    trace = external_trace(_minimal_values())
    assert trace.producer == EXTERNAL_PRODUCER


def test_external_trace_carries_values() -> None:
    art = TextArtifact(text="in")
    trace = external_trace(
        {INPUT_KEY: art, OUTPUT_KEY: TextArtifact(text="out")}
    )
    assert trace.value(INPUT_KEY) == art


def test_external_trace_validates_value_types() -> None:
    # external_trace validates value types on the way in.
    with pytest.raises((WiringError, ValueError, TypeError)):
        external_trace(
            {
                INPUT_KEY: TextArtifact(text="in"),
                OUTPUT_KEY: "not a trace value",  # type: ignore[dict-item]
            }
        )


def test_external_trace_accepts_step_facts() -> None:
    trace = external_trace(
        _minimal_values(),
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


# --- producer identity -----------------------------------------------


def test_trace_records_non_external_producer() -> None:
    producer = TraceProducer(producer_id="preproc-1", version="1.0")
    trace = Trace(values=_minimal_values(), producer=producer)
    assert trace.producer.producer_id == "preproc-1"
