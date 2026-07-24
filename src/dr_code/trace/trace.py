"""Trace, reserved keys, WiringError."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, get_args

from pydantic import JsonValue

from dr_code.trace.absent import Absent
from dr_code.trace.artifacts import Artifact
from dr_code.trace.facts import validate_step_facts
from dr_code.trace.provenance import EXTERNAL_PRODUCER, TraceProducer

INPUT_KEY: Final = "input"
OUTPUT_KEY: Final = "output"
RESERVED_KEYS: Final = frozenset({INPUT_KEY, OUTPUT_KEY})

TraceValue = Artifact | Absent

# Concrete runtime classes that a TraceValue may be, derived from the
# Artifact union metadata plus Absent. get_args(Artifact) unwraps the
# Annotated[Union[...], Field] form to the member model classes.
_TRACE_VALUE_TYPES: Final = (*get_args(get_args(Artifact)[0]), Absent)


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
    step_facts: Mapping[str, Mapping[str, JsonValue]] = field(
        default_factory=dict
    )
    # step_facts: provenance recorded by steps (chosen alternative,
    # rejection reasons, candidate counts) keyed by instance name —
    # facts, never judgments (P-L2)

    def __post_init__(self) -> None:
        """Validate RESERVED_KEYS ⊆ values; reject non-TraceValue
        entries."""
        missing = RESERVED_KEYS - self.values.keys()
        if missing:
            raise WiringError(
                "trace missing reserved key(s): " + ", ".join(sorted(missing))
            )
        for key, val in self.values.items():
            if not isinstance(val, _TRACE_VALUE_TYPES):
                raise WiringError(
                    f"value for key {key!r} is not a TraceValue: "
                    f"{type(val).__name__}"
                )
        validate_step_facts(self.step_facts)

    def value(self, key: str) -> TraceValue:
        """Missing key raises WiringError. Present-but-Absent returns the
        Absent value — callers decide what not-applicable means for them.
        """
        try:
            return self.values[key]
        except KeyError:
            raise WiringError(f"trace has no value for key {key!r}") from None


def external_trace(
    values: Mapping[str, TraceValue],
    *,
    step_facts: Mapping[str, Mapping[str, JsonValue]] | None = None,
) -> Trace:
    """Boundary constructor for artifacts built outside dr-code:
    validates value types on the way in, stamps
    producer=EXTERNAL_PRODUCER (X-S2).
    """
    return Trace(
        values=dict(values),
        producer=EXTERNAL_PRODUCER,
        step_facts=dict(step_facts) if step_facts is not None else {},
    )
