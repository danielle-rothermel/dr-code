"""SerializedTrace + round-trip functions."""

from __future__ import annotations

from typing import Final

from dr_code.models import FrozenModel
from dr_code.trace.absent import Absent
from dr_code.trace.artifacts import Artifact
from dr_code.trace.provenance import TraceProducer
from dr_code.trace.trace import Trace

TRACE_SCHEMA_VERSION: Final = 1


class SerializedTrace(FrozenModel):
    """Canonical artifacts, names, causal absences, and provenance — no
    derived views (eval-flow L3). BaseModel so it feeds persistence and
    external schemas.
    """

    schema_version: int = TRACE_SCHEMA_VERSION
    producer: TraceProducer
    values: dict[str, Artifact | Absent]
    step_facts: dict[str, dict[str, str]] = {}


def serialize_trace(trace: Trace) -> SerializedTrace:
    """Total for traces built from TraceValue types. May lose warm
    caches, never information; round-trip must be value-equal (S3).
    """
    raise NotImplementedError


def deserialize_trace(serialized: SerializedTrace) -> Trace:
    """Restored traces have cold caches; measuring one later must equal
    measuring the fresh trace now — enforced by a metrics-side test.
    """
    raise NotImplementedError
