"""SerializedTrace + round-trip functions.

Deserialization is deliberately permissive: ``deserialize_trace`` and the
trace models validate structure only — schema version, value shapes, and
producer discriminators — and never check producer coordinates against the
live component registries. Archived traces stay loadable after the
registries move on; semantic validity against the current pipeline is
enforced at use time by the runner's resolve-and-compare guard.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import field_validator

from dr_code.base import FrozenModel
from dr_code.trace.facts import JsonFactValue, validate_step_facts
from dr_code.trace.provenance import TraceProducer
from dr_code.trace.trace import Trace, TraceValue

TRACE_SCHEMA_VERSION: Final = 3


class SerializedTrace(FrozenModel):
    """Canonical artifacts, names, causal absences, and provenance — no
    derived views. This boundary model supports persistence and external
    schemas.
    """

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
        """Run the finite-JSON gate at the persistence boundary too.

        Pydantic's ``JsonFactValue`` union admits non-finite floats, which
        have no JSON form; ``validate_step_facts`` rejects them and copies
        the containers so the model owns its facts.
        """
        return validate_step_facts(step_facts)


def serialize_trace(trace: Trace) -> SerializedTrace:
    """Total for traces built from TraceValue types. May lose warm
    caches, never canonical values; round-trips remain value-equal.
    """
    return SerializedTrace(
        schema_version=TRACE_SCHEMA_VERSION,
        producer=trace.producer,
        values=dict(trace.values),
        step_facts=validate_step_facts(trace.step_facts),
    )


def deserialize_trace(serialized: SerializedTrace) -> Trace:
    """Restored traces have cold caches; measuring one later must equal
    measuring the fresh trace now — enforced by a metrics-side test.
    """
    return Trace(
        values=serialized.values,
        producer=serialized.producer,
        step_facts=serialized.step_facts,
    )
