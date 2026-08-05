"""Trace, reserved keys, WiringError."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, get_args

from dr_code.trace.absent import Absent
from dr_code.trace.artifacts import Artifact, JsonArtifact
from dr_code.trace.facts import (
    FactError,
    JsonFactValue,
    validate_step_facts,
)
from dr_code.trace.provenance import (
    EXTERNAL_PRODUCER,
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

# Concrete producer classes a Trace may be stamped with, derived from the
# TraceProducer union metadata the same way.
_TRACE_PRODUCER_TYPES: Final = get_args(get_args(TraceProducer)[0])


class WiringError(Exception):
    """Incompatible definitions: missing key, wrong artifact kind, or
    invalid settings. Trace construction, value lookup, binding, and execution
    raise this error when they encounter incompatible wiring rather than
    recording it as a per-input outcome.
    """


def _snapshot_value(value: TraceValue) -> TraceValue:
    """The value as the trace will hold it, independent of the caller.

    Every trace value is a frozen model, but freezing only bars attribute
    assignment: it does not stop a caller from mutating a container the
    model still points at. ``JsonArtifact.payload`` is the one trace value
    holding arbitrary nested ``dict``/``list`` data, so it is the only one
    that needs copying — every other artifact carries strings, tuples of
    frozen models, or scalars, none of which can be mutated in place.
    """

    if isinstance(value, JsonArtifact):
        return JsonArtifact(payload=copy.deepcopy(value.payload))
    return value


@dataclass(frozen=True, slots=True)
class Trace:
    """A snapshot: constructing a trace copies the containers it is given.

    ``values`` is shallow-copied into a plain dict and ``step_facts`` is
    deep-copied by ``validate_step_facts``, so mutating the caller's
    mappings afterwards cannot change an existing trace. The individual
    values are frozen models whose own containers are copied at validation
    — ``JsonArtifact.payload`` included — so mutating a payload a caller
    still holds cannot change the artifact recorded in the trace.
    """

    # flat namespace; must contain input & output
    values: Mapping[str, TraceValue]
    producer: TraceProducer
    step_facts: Mapping[str, Mapping[str, JsonFactValue]] = field(
        default_factory=dict
    )
    # step_facts: provenance recorded by steps (chosen alternative,
    # rejection reasons, candidate counts) keyed by instance name —
    # descriptive facts, never policy judgments

    def __post_init__(self) -> None:
        """Snapshot the containers, then validate keys, values, and facts."""
        if not isinstance(self.producer, _TRACE_PRODUCER_TYPES):
            raise WiringError(
                "trace producer must be an external or preprocessing "
                "producer coordinate"
            )
        missing = RESERVED_KEYS - self.values.keys()
        if missing:
            raise WiringError(
                "trace missing reserved key(s): " + ", ".join(sorted(missing))
            )
        for key, val in self.values.items():
            if not isinstance(key, str):
                raise WiringError(
                    f"trace value keys must be strings: {type(key).__name__}"
                )
            if not isinstance(val, _TRACE_VALUE_TYPES):
                raise WiringError(
                    f"value for key {key!r} is not a TraceValue: "
                    f"{type(val).__name__}"
                )
        try:
            snapshot_facts = validate_step_facts(self.step_facts)
        except FactError as exc:
            raise WiringError(f"invalid step facts: {exc}") from exc
        object.__setattr__(
            self,
            "values",
            {key: _snapshot_value(val) for key, val in self.values.items()},
        )
        object.__setattr__(self, "step_facts", snapshot_facts)

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
    step_facts: Mapping[str, Mapping[str, JsonFactValue]] | None = None,
) -> Trace:
    """Boundary constructor for artifacts built outside dr-code:
    validates value types on the way in and stamps the external producer.
    """
    return Trace(
        values=values,
        producer=EXTERNAL_PRODUCER,
        step_facts=step_facts if step_facts is not None else {},
    )
