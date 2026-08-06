from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from typing import cast

import pytest

from dr_code.trace import (
    EXTERNAL_PRODUCER,
    INPUT_KEY,
    OUTPUT_KEY,
    Absent,
    JsonArtifact,
    TextArtifact,
    Trace,
    TraceValue,
    WiringError,
    external_trace,
)


def _minimal_values() -> dict[str, TraceValue]:
    return {
        INPUT_KEY: TextArtifact(text="input"),
        OUTPUT_KEY: TextArtifact(text="output"),
    }


def _attempt_public_mutation(action: Callable[[], object]) -> None:
    try:
        action()
    except (AttributeError, TypeError):
        pass


def test_trace_requires_both_reserved_values() -> None:
    with pytest.raises(
        WiringError,
        match=r"trace missing reserved key\(s\): output",
    ):
        Trace(
            values={INPUT_KEY: TextArtifact(text="input")},
            producer=EXTERNAL_PRODUCER,
        )


def test_trace_rejects_non_trace_values() -> None:
    values: dict[str, object] = _minimal_values()
    values["invalid"] = "not a trace value"

    with pytest.raises(
        WiringError,
        match="value for key 'invalid' is not a TraceValue: str",
    ):
        Trace(values=values, producer=EXTERNAL_PRODUCER)  # type: ignore[arg-type]


def test_trace_rejects_non_string_keys() -> None:
    values = {**_minimal_values(), 1: TextArtifact(text="integer key")}

    with pytest.raises(WiringError):
        Trace(values=values, producer=EXTERNAL_PRODUCER)  # type: ignore[arg-type]


def test_trace_rejects_unvalidated_producer_payload() -> None:
    with pytest.raises(WiringError, match="trace producer"):
        Trace(
            values=_minimal_values(),
            producer={"kind": "external"},  # type: ignore[arg-type]
        )


def test_value_distinguishes_causal_absence_from_missing_wiring() -> None:
    absent = Absent(
        failed_step="parse",
        failure_code="parse_failed",
        cause="syntax error",
    )
    trace = Trace(
        values={
            INPUT_KEY: TextArtifact(text="input"),
            OUTPUT_KEY: absent,
        },
        producer=EXTERNAL_PRODUCER,
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

    trace = external_trace(values, step_facts=step_facts)

    assert trace.producer == EXTERNAL_PRODUCER
    assert trace.value(INPUT_KEY) is values[INPUT_KEY]
    assert trace.step_facts == step_facts


def test_trace_snapshots_values_against_later_caller_mutation() -> None:
    values = _minimal_values()
    trace = Trace(values=values, producer=EXTERNAL_PRODUCER)

    values["late"] = TextArtifact(text="added after construction")
    del values[INPUT_KEY]

    assert set(trace.values) == {INPUT_KEY, OUTPUT_KEY}
    assert trace.value(INPUT_KEY) == TextArtifact(text="input")


def test_trace_snapshots_json_payloads_against_later_caller_mutation() -> None:
    # Freeze is shallow; copy nested JSON so callers cannot rewrite snapshots.
    payload = {"task_id": "HumanEval/0", "nested": {"names": ["a"]}}
    artifact = JsonArtifact(payload=payload)
    values = _minimal_values()
    values["task"] = artifact
    trace = Trace(values=values, producer=EXTERNAL_PRODUCER)

    payload["task_id"] = "mutated after construction"
    payload["nested"]["names"].append("b")
    artifact.payload["nested"]["names"].append("c")

    assert trace.value("task") == JsonArtifact(
        payload={"task_id": "HumanEval/0", "nested": {"names": ["a"]}}
    )


def test_trace_values_public_view_cannot_change_snapshot() -> None:
    trace = Trace(values=_minimal_values(), producer=EXTERNAL_PRODUCER)

    _attempt_public_mutation(
        lambda: operator.setitem(
            trace.values,
            "late",
            TextArtifact(text="added through public view"),
        )
    )
    _attempt_public_mutation(
        lambda: operator.setitem(
            trace.values,
            INPUT_KEY,
            TextArtifact(text="replaced through public view"),
        )
    )

    assert set(trace.values) == {INPUT_KEY, OUTPUT_KEY}
    assert trace.value(INPUT_KEY) == TextArtifact(text="input")


def test_trace_value_public_json_payload_cannot_change_snapshot() -> None:
    values = _minimal_values()
    values["task"] = JsonArtifact(payload={"nested": {"names": ["a"]}})
    trace = Trace(values=values, producer=EXTERNAL_PRODUCER)

    observed = trace.value("task")
    assert isinstance(observed, JsonArtifact)
    payload = cast(Mapping[str, object], observed.payload)
    nested = cast(Mapping[str, object], payload["nested"])
    names = nested["names"]
    _attempt_public_mutation(
        lambda: cast(list[object], names).append("mutated through public view")
    )

    assert trace.value("task") == JsonArtifact(
        payload={"nested": {"names": ["a"]}}
    )
