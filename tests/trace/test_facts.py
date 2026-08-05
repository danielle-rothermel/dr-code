"""Validation and snapshot tests for trace step facts."""

from __future__ import annotations

from array import array
from collections.abc import Callable, Mapping
from enum import IntEnum, StrEnum
from typing import cast

import pytest

from dr_code.trace import (
    EXTERNAL_PRODUCER,
    INPUT_KEY,
    OUTPUT_KEY,
    TextArtifact,
    Trace,
    TraceValue,
    WiringError,
)


def _minimal_values() -> dict[str, TraceValue]:
    return {
        INPUT_KEY: TextArtifact(text="input"),
        OUTPUT_KEY: TextArtifact(text="output"),
    }


def _attempt_public_mutation(action: Callable[[], object]) -> None:
    """Exercise either a mutable defensive copy or an immutable view."""
    try:
        action()
    except (AttributeError, TypeError):
        pass


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


def test_trace_step_facts_public_view_cannot_change_snapshot() -> None:
    trace = Trace(
        values=_minimal_values(),
        producer=EXTERNAL_PRODUCER,
        step_facts={
            "parse": {"detail": {"rejected_locations": [0]}},
        },
    )

    detail = cast(Mapping[str, object], trace.step_facts["parse"]["detail"])
    rejected_locations = detail["rejected_locations"]
    _attempt_public_mutation(
        lambda: cast(list[object], rejected_locations).append(1)
    )

    assert trace.step_facts == {
        "parse": {"detail": {"rejected_locations": [0]}}
    }


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
