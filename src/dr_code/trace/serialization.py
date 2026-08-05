from __future__ import annotations

from typing import Final, Literal

from pydantic import field_validator

from dr_code.core.models import FrozenModel
from dr_code.trace.facts import JsonFactValue, validate_step_facts
from dr_code.trace.provenance import TraceProducer
from dr_code.trace.trace import Trace, TraceValue

TRACE_SCHEMA_VERSION: Final = 3


class SerializedTrace(FrozenModel):
    schema_version: Literal[3]
    producer: TraceProducer
    values: dict[str, TraceValue]
    step_facts: dict[str, dict[str, JsonFactValue]] = {}

    @field_validator("step_facts")
    @classmethod
    def _check_step_facts(
        cls,
        step_facts: dict[str, dict[str, JsonFactValue]],
    ) -> dict[str, dict[str, JsonFactValue]]:
        return validate_step_facts(step_facts)


def serialize_trace(trace: Trace) -> SerializedTrace:
    return SerializedTrace(
        schema_version=TRACE_SCHEMA_VERSION,
        producer=trace.producer,
        values=dict(trace.values),
        step_facts=validate_step_facts(trace.step_facts),
    )


def deserialize_trace(serialized: SerializedTrace) -> Trace:
    return Trace(
        values=serialized.values,
        producer=serialized.producer,
        step_facts=serialized.step_facts,
    )
