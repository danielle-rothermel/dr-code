from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, get_args

from dr_code.trace.absent import Absent
from dr_code.trace.artifacts import Artifact
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

_EMPTY_STEP_FACTS: Final[Mapping[str, Mapping[str, JsonFactValue]]] = (
    MappingProxyType({})
)

_TRACE_VALUE_TYPES: Final = (*get_args(get_args(Artifact)[0]), Absent)

_TRACE_PRODUCER_TYPES: Final = get_args(get_args(TraceProducer)[0])


class WiringError(Exception):
    pass


def _snapshot_value(value: TraceValue) -> TraceValue:
    # Artifact models recursively freeze their nested public containers.
    return value


@dataclass(frozen=True, slots=True, init=False)
class Trace:
    _values: dict[str, TraceValue] = field(repr=False)
    producer: TraceProducer
    _step_facts: dict[str, dict[str, JsonFactValue]] = field(repr=False)

    def __init__(
        self,
        values: Mapping[str, TraceValue],
        producer: TraceProducer,
        step_facts: Mapping[
            str, Mapping[str, JsonFactValue]
        ] = _EMPTY_STEP_FACTS,
    ) -> None:
        if not isinstance(producer, _TRACE_PRODUCER_TYPES):
            raise WiringError(
                "trace producer must be an external or preprocessing "
                "producer coordinate"
            )
        missing = RESERVED_KEYS - values.keys()
        if missing:
            raise WiringError(
                "trace missing reserved key(s): " + ", ".join(sorted(missing))
            )
        for key, val in values.items():
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
            snapshot_facts = validate_step_facts(step_facts)
        except FactError as exc:
            raise WiringError(f"invalid step facts: {exc}") from exc
        object.__setattr__(
            self,
            "_values",
            {key: _snapshot_value(val) for key, val in values.items()},
        )
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "_step_facts", snapshot_facts)

    @property
    def values(self) -> Mapping[str, TraceValue]:
        return {
            key: _snapshot_value(value) for key, value in self._values.items()
        }

    @property
    def step_facts(self) -> Mapping[str, Mapping[str, JsonFactValue]]:
        return validate_step_facts(self._step_facts)

    def value(self, key: str) -> TraceValue:
        try:
            return _snapshot_value(self._values[key])
        except KeyError:
            raise WiringError(f"trace has no value for key {key!r}") from None


def external_trace(
    values: Mapping[str, TraceValue],
    *,
    step_facts: Mapping[str, Mapping[str, JsonFactValue]] | None = None,
) -> Trace:
    return Trace(
        values=values,
        producer=EXTERNAL_PRODUCER,
        step_facts=step_facts if step_facts is not None else {},
    )
