"""SerializedTrace + round-trip functions."""

from __future__ import annotations

from typing import Final, Literal

from dr_code.models import FrozenModel
from dr_code.trace.absent import Absent
from dr_code.trace.artifacts import Artifact
from dr_code.trace.provenance import TraceProducer
from dr_code.trace.trace import Trace
from dr_code.trace.observation import SampleIdentity

TRACE_SCHEMA_VERSION: Final = 3


class SerializedTrace(FrozenModel):
    """Canonical artifacts, names, causal absences, and provenance — no
    derived views. This boundary model supports persistence and external
    schemas.
    """

    schema_version: Literal[3]
    producer: TraceProducer
    sample_identity: SampleIdentity | None = None
    values: dict[str, Artifact | Absent]
    step_facts: dict[str, dict[str, str]] = {}


def serialize_trace(trace: Trace) -> SerializedTrace:
    """Total for traces built from TraceValue types. May lose warm
    caches, never canonical values; round-trips remain value-equal.
    """
    return SerializedTrace(
        schema_version=TRACE_SCHEMA_VERSION,
        producer=trace.producer,
        sample_identity=trace.sample_identity,
        values=dict(trace.values),
        step_facts={k: dict(v) for k, v in trace.step_facts.items()},
    )


def deserialize_trace(serialized: SerializedTrace) -> Trace:
    """Restored traces have cold caches; measuring one later must equal
    measuring the fresh trace now — enforced by a metrics-side test.
    """
    if serialized.schema_version != TRACE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported serialized trace schema version: "
            f"{serialized.schema_version}"
        )
    return Trace(
        values=dict(serialized.values),
        producer=serialized.producer,
        sample_identity=serialized.sample_identity,
        step_facts={k: dict(v) for k, v in serialized.step_facts.items()},
    )
