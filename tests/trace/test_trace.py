"""Domain tests for trace wiring."""

from __future__ import annotations

from array import array
from enum import IntEnum, StrEnum

import pytest

from dr_code.trace import (
    EXTERNAL_PRODUCER,
    INPUT_KEY,
    OUTPUT_KEY,
    Absent,
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


# --- snapshotting ----------------------------------------------------


def test_trace_snapshots_values_against_later_caller_mutation() -> None:
    values = _minimal_values()
    trace = Trace(values=values, producer=EXTERNAL_PRODUCER)

    values["late"] = TextArtifact(text="added after construction")
    del values[INPUT_KEY]

    assert set(trace.values) == {INPUT_KEY, OUTPUT_KEY}
    assert trace.value(INPUT_KEY) == TextArtifact(text="input")


def test_trace_deep_copies_step_facts_against_later_mutation() -> None:
    nested = {"rejected_locations": [0]}
    step_facts = {"parse": {"detail": nested}}
    trace = Trace(
        values=_minimal_values(),
        producer=EXTERNAL_PRODUCER,
        step_facts=step_facts,
    )

    nested["rejected_locations"].append(1)
    step_facts["parse"]["extra"] = "added after construction"
    step_facts["late"] = {"reason": "added after construction"}

    assert trace.step_facts == {
        "parse": {"detail": {"rejected_locations": [0]}}
    }


# --- step fact validation --------------------------------------------


def test_trace_accepts_finite_json_step_facts() -> None:
    step_facts = {
        "parse": {
            "alternative": "fenced_blocks",
            "candidate_count": 2,
            "confidence": 0.25,
            "recoverable": False,
            "hint": None,
            "rejected_locations": [0, 1],
            "detail": {"line": 3},
        }
    }

    trace = Trace(
        values=_minimal_values(),
        producer=EXTERNAL_PRODUCER,
        step_facts=step_facts,
    )

    assert trace.step_facts == step_facts


@pytest.mark.parametrize(
    "facts",
    (
        {"parse": {"confidence": float("nan")}},
        {"parse": {"confidence": float("inf")}},
        {"parse": {"confidence": float("-inf")}},
        {"parse": {"value": {1: "non-string key"}}},
        {"parse": {"value": object()}},
        {"parse": {"value": {"nested": object()}}},
        {"parse": "not a mapping"},
        # The bytes family is Sequence-shaped but has no JSON form; it must
        # be rejected rather than coerced into a list of ints.
        {"parse": {"value": b"raw"}},
        {"parse": {"value": bytearray(b"raw")}},
        {"parse": {"value": memoryview(b"raw")}},
        {"parse": {"value": array("i", [1, 2])}},
        {"parse": {"value": {"nested": b"raw"}}},
    ),
)
def test_trace_rejects_non_json_step_facts(facts: object) -> None:
    with pytest.raises(WiringError, match="invalid step facts"):
        Trace(
            values=_minimal_values(),
            producer=EXTERNAL_PRODUCER,
            step_facts=facts,  # type: ignore[arg-type]
        )


def test_trace_narrows_enum_step_fact_leaves_to_plain_builtins() -> None:
    class Alternative(StrEnum):
        FENCED_BLOCKS = "fenced_blocks"

    class Attempts(IntEnum):
        TWO = 2

    trace = Trace(
        values=_minimal_values(),
        producer=EXTERNAL_PRODUCER,
        step_facts={
            "parse": {
                "alternative": Alternative.FENCED_BLOCKS,
                "attempts": Attempts.TWO,
                "nested": {"alternative": Alternative.FENCED_BLOCKS},
            }
        },
    )

    stored = trace.step_facts["parse"]
    assert stored == {
        "alternative": "fenced_blocks",
        "attempts": 2,
        "nested": {"alternative": "fenced_blocks"},
    }
    # Equality alone would pass for the live enum members; the stored facts
    # must hold plain containers, so pin the exact leaf types.
    assert type(stored["alternative"]) is str
    assert type(stored["attempts"]) is int
    nested = stored["nested"]
    assert isinstance(nested, dict)
    assert type(nested["alternative"]) is str


def test_trace_rejects_step_facts_with_a_container_cycle() -> None:
    cycle: list[object] = []
    cycle.append(cycle)

    with pytest.raises(WiringError, match="container cycle"):
        Trace(
            values=_minimal_values(),
            producer=EXTERNAL_PRODUCER,
            step_facts={"parse": {"value": cycle}},  # type: ignore[dict-item]
        )
