"""Domain tests for trace wiring."""

from __future__ import annotations

import pytest

from dr_code.trace import (
    EXTERNAL_PRODUCER_ID,
    INPUT_KEY,
    OUTPUT_KEY,
    Absent,
    ExternalSource,
    TextArtifact,
    Trace,
    TraceValue,
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


def _minimal_values() -> dict[str, TraceValue]:
    return {
        INPUT_KEY: TextArtifact(text="input"),
        OUTPUT_KEY: TextArtifact(text="output"),
    }


def test_trace_requires_both_reserved_values() -> None:
    with pytest.raises(
        WiringError,
        match=r"trace missing reserved key\(s\): output",
    ):
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


def test_value_distinguishes_causal_absence_from_missing_wiring() -> None:
    absent = Absent(failed_step="parse", cause="syntax error")
    trace = Trace(
        values={
            INPUT_KEY: TextArtifact(text="input"),
            OUTPUT_KEY: absent,
        },
        producer=_producer(),
    )

    assert trace.value(OUTPUT_KEY) is absent
    with pytest.raises(
        WiringError,
        match="trace has no value for key 'unknown'",
    ):
        trace.value("unknown")


def test_external_trace_stamps_producer_and_carries_boundary_data() -> None:
    values = _minimal_values()
    step_facts = {"parse": {"choice": "candidate_0"}}

    trace = external_trace(values, source=_SOURCE, step_facts=step_facts)

    assert trace.producer == _producer()
    assert trace.value(INPUT_KEY) is values[INPUT_KEY]
    assert trace.step_facts == step_facts
