"""Trace, reserved keys, WiringError."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

from dr_code.trace.absent import Absent
from dr_code.trace.artifacts import Artifact
from dr_code.trace.provenance import TraceProducer

INPUT_KEY: Final = "input"
OUTPUT_KEY: Final = "output"
RESERVED_KEYS: Final = frozenset({INPUT_KEY, OUTPUT_KEY})

TraceValue = Artifact | Absent


class WiringError(Exception):
    """Incompatible definitions: missing key, wrong artifact kind, or
    invalid settings. Raised at bind time — before any input is
    processed — and never per-input (eval-flow L2). Both systems raise
    this same type.
    """


@dataclass(frozen=True, slots=True)
class Trace:
    # flat namespace; must contain input & output
    values: Mapping[str, TraceValue]
    producer: TraceProducer
    step_facts: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    # step_facts: provenance recorded by steps (chosen alternative,
    # rejection reasons, candidate counts) keyed by instance name —
    # facts, never judgments (P-L2)

    def __post_init__(self) -> None:
        """Validate RESERVED_KEYS ⊆ values; reject non-TraceValue
        entries."""
        raise NotImplementedError

    def value(self, key: str) -> TraceValue:
        """Missing key raises WiringError. Present-but-Absent returns the
        Absent value — callers decide what not-applicable means for them.
        """
        raise NotImplementedError


def external_trace(
    values: Mapping[str, TraceValue],
    *,
    step_facts: Mapping[str, Mapping[str, str]] | None = None,
) -> Trace:
    """Boundary constructor for artifacts built outside dr-code:
    validates value types on the way in, stamps
    producer=EXTERNAL_PRODUCER (X-S2).
    """
    raise NotImplementedError
