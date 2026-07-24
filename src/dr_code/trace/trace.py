"""Trace, reserved keys, WiringError."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, get_args

from dr_code.trace.absent import Absent
from dr_code.trace.artifacts import Artifact
from dr_code.trace.observation import SampleIdentity
from dr_code.trace.provenance import (
    EXTERNAL_PRODUCER_ID,
    ExternalSource,
    TraceProducer,
)

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
    invalid settings. Trace construction, value lookup, binding, and execution
    raise this error when they encounter incompatible wiring rather than
    recording it as a per-input outcome.
    """


@dataclass(frozen=True, slots=True)
class Trace:
    # flat namespace; must contain input & output
    values: Mapping[str, TraceValue]
    producer: TraceProducer
    step_facts: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    sample_identity: SampleIdentity | None = None
    # step_facts: provenance recorded by steps (chosen alternative,
    # rejection reasons, candidate counts) keyed by instance name —
    # descriptive facts, never policy judgments

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
    source: ExternalSource,
    step_facts: Mapping[str, Mapping[str, str]] | None = None,
    sample_identity: SampleIdentity | None = None,
) -> Trace:
    """Boundary constructor for artifacts built outside dr-code.

    Validates value types on the way in and stamps the producer with the
    caller's authenticated external source.
    """
    return Trace(
        values=dict(values),
        producer=TraceProducer(
            producer_id=EXTERNAL_PRODUCER_ID,
            external_source=source,
        ),
        step_facts=dict(step_facts) if step_facts is not None else {},
        sample_identity=sample_identity,
    )
