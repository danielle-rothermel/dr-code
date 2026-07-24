"""Trace, reserved keys, WiringError."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, get_args

from pydantic import JsonValue

from dr_code.trace.absent import Absent
from dr_code.trace.artifacts import Artifact
from dr_code.trace.facts import validate_step_facts
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


@dataclass(frozen=True, slots=True, init=False)
class Trace:
    _values: Mapping[str, TraceValue] = field(repr=False)
    producer: TraceProducer
    _step_facts: Mapping[str, Mapping[str, JsonValue]] = field(repr=False)
    sample_identity: SampleIdentity | None = None
    # step_facts: provenance recorded by steps (chosen alternative,
    # rejection reasons, candidate counts) keyed by instance name —
    # descriptive facts, never policy judgments

    def __init__(
        self,
        values: Mapping[str, TraceValue],
        producer: TraceProducer,
        step_facts: Mapping[str, Mapping[str, JsonValue]] | None = None,
        sample_identity: SampleIdentity | None = None,
    ) -> None:
        """Validate and snapshot a complete trace boundary."""
        missing = RESERVED_KEYS - values.keys()
        if missing:
            raise WiringError(
                "trace missing reserved key(s): " + ", ".join(sorted(missing))
            )
        for key, val in values.items():
            if not isinstance(val, _TRACE_VALUE_TYPES):
                raise WiringError(
                    f"value for key {key!r} is not a TraceValue: "
                    f"{type(val).__name__}"
                )
        facts = step_facts if step_facts is not None else {}
        validate_step_facts(facts)
        object.__setattr__(
            self,
            "_values",
            MappingProxyType(deepcopy(dict(values))),
        )
        object.__setattr__(self, "sample_identity", sample_identity)
        object.__setattr__(self, "producer", producer)
        object.__setattr__(
            self,
            "_step_facts",
            MappingProxyType(deepcopy(dict(facts))),
        )

    @property
    def values(self) -> Mapping[str, TraceValue]:
        """Return a defensive copy of the flat trace namespace."""
        return MappingProxyType(deepcopy(dict(self._values)))

    @property
    def step_facts(self) -> Mapping[str, Mapping[str, JsonValue]]:
        """Return defensive copies of JSON-valued per-step facts."""
        return MappingProxyType(deepcopy(dict(self._step_facts)))

    def value(self, key: str) -> TraceValue:
        """Missing key raises WiringError. Present-but-Absent returns the
        Absent value — callers decide what not-applicable means for them.
        """
        try:
            return deepcopy(self._values[key])
        except KeyError:
            raise WiringError(f"trace has no value for key {key!r}") from None


def external_trace(
    values: Mapping[str, TraceValue],
    *,
    source: ExternalSource,
    step_facts: Mapping[str, Mapping[str, JsonValue]] | None = None,
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
