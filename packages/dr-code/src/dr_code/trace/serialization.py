from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal, cast

from pydantic import Field, field_serializer, field_validator

from dr_code.core.models import FrozenModel
from dr_code.trace.artifacts import freeze_json_value, json_value_to_wire
from dr_code.trace.facts import JsonFactValue, validate_step_facts
from dr_code.trace.provenance import TraceProducer
from dr_code.trace.trace import Trace, TraceValue

TRACE_SCHEMA_VERSION: Final = 3


class SerializedTrace(FrozenModel):
    schema_version: Literal[3]
    producer: TraceProducer
    values: Mapping[str, TraceValue]
    step_facts: Mapping[str, Mapping[str, JsonFactValue]] = Field(
        default_factory=dict
    )

    @field_validator("values")
    @classmethod
    def _freeze_values(
        cls, values: Mapping[str, TraceValue]
    ) -> Mapping[str, TraceValue]:
        return MappingProxyType(dict(values))

    @field_validator("step_facts")
    @classmethod
    def _check_step_facts(
        cls,
        step_facts: Mapping[str, Mapping[str, JsonFactValue]],
    ) -> Mapping[str, Mapping[str, JsonFactValue]]:
        validated = validate_step_facts(step_facts)
        return MappingProxyType(
            {
                instance_name: MappingProxyType(
                    {
                        key: freeze_json_value(value)
                        for key, value in facts.items()
                    }
                )
                for instance_name, facts in validated.items()
            }
        )

    @field_serializer("values")
    def _serialize_values(
        self, values: Mapping[str, TraceValue]
    ) -> dict[str, TraceValue]:
        return dict(values)

    @field_serializer("step_facts")
    def _serialize_step_facts(
        self,
        step_facts: Mapping[str, Mapping[str, JsonFactValue]],
    ) -> dict[str, dict[str, JsonFactValue]]:
        return {
            instance_name: {
                key: cast(JsonFactValue, json_value_to_wire(value))
                for key, value in facts.items()
            }
            for instance_name, facts in step_facts.items()
        }


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
