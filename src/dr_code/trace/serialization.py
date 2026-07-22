"""SerializedTrace + round-trip functions."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import JsonValue, model_validator

from dr_code.models import FrozenModel
from dr_code.trace.absent import Absent
from dr_code.trace.artifacts import Artifact
from dr_code.trace.facts import validate_step_facts
from dr_code.trace.provenance import TraceProducer
from dr_code.trace.trace import Trace

TRACE_SCHEMA_VERSION: Final = 3


class SerializedTrace(FrozenModel):
    """Canonical artifacts, names, causal absences, and provenance — no
    derived views (eval-flow L3). BaseModel so it feeds persistence and
    external schemas.
    """

    schema_version: Literal[3]
    producer: TraceProducer
    values: dict[str, Artifact | Absent]
    step_facts: dict[str, dict[str, JsonValue]] = {}

    @model_validator(mode="after")
    def _validate_facts_are_json_lossless(self) -> SerializedTrace:
        validate_step_facts(self.step_facts)
        return self


def serialize_trace(trace: Trace) -> SerializedTrace:
    """Total for traces built from TraceValue types. May lose warm
    caches, never information; round-trip must be value-equal (S3).
    """
    return SerializedTrace(
        schema_version=TRACE_SCHEMA_VERSION,
        producer=trace.producer,
        values=dict(trace.values),
        step_facts={k: dict(v) for k, v in trace.step_facts.items()},
    )


def deserialize_trace(serialized: SerializedTrace) -> Trace:
    """Restored traces have cold caches; measuring one later must equal
    measuring the fresh trace now — enforced by a metrics-side test.
    """
    return Trace(
        values=dict(serialized.values),
        producer=serialized.producer,
        step_facts={k: dict(v) for k, v in serialized.step_facts.items()},
    )
