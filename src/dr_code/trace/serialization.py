"""SerializedTrace + round-trip functions.

Deserialization is deliberately permissive: ``deserialize_trace`` and the
trace models validate structure only — schema version, value shapes, and
producer discriminators — and never check producer coordinates against the
live component registries. Previously stored traces stay loadable after the
registries move on; semantic validity against the current pipeline is
enforced at use time by the runner's resolve-and-compare guard.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import JsonValue

from dr_code.models import FrozenModel
from dr_code.trace.absent import Absent
from dr_code.trace.artifacts import Artifact
from dr_code.trace.provenance import TraceProducer
from dr_code.trace.trace import Trace

TRACE_SCHEMA_VERSION: Final = 2


class SerializedTrace(FrozenModel):
    """Canonical artifacts, names, causal absences, and provenance — no
    derived views. This boundary model supports persistence and external
    schemas.
    """

    schema_version: Literal[2]
    producer: TraceProducer
    values: dict[str, Artifact | Absent]
    step_facts: dict[str, dict[str, JsonValue]] = {}


def serialize_trace(trace: Trace) -> SerializedTrace:
    """Total for traces built from TraceValue types. May lose warm
    caches, never canonical values; round-trips remain value-equal.
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
