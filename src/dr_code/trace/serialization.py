"""SerializedTrace + round-trip functions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Literal

from pydantic import JsonValue, model_validator

from dr_code.models import FrozenModel
from dr_code.trace.absent import Absent
from dr_code.trace.artifacts import Artifact
from dr_code.trace.facts import reject_nonfinite_floats
from dr_code.trace.provenance import TraceProducer
from dr_code.trace.trace import Trace

TRACE_SCHEMA_VERSION: Final = 2


class SerializedTrace(FrozenModel):
    """Canonical artifacts, names, causal absences, and provenance — no
    derived views (eval-flow L3). BaseModel so it feeds persistence and
    external schemas.
    """

    # Version 1 payloads are upgraded during validation. Model instances are
    # always canonical v2, so dumping one can never emit a hybrid v1 envelope.
    schema_version: Literal[2] = TRACE_SCHEMA_VERSION
    producer: TraceProducer
    values: dict[str, Artifact | Absent]
    step_facts: dict[str, dict[str, JsonValue]] = {}

    @model_validator(mode="before")
    @classmethod
    def _upgrade_v1_and_validate_v2(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value

        payload = dict(value)
        version = payload.get("schema_version", TRACE_SCHEMA_VERSION)
        values = payload.get("values")
        if not isinstance(values, Mapping):
            return payload

        migrated_values: dict[str, Any] = {}
        for key, trace_value in values.items():
            if isinstance(trace_value, Absent):
                migrated_values[key] = trace_value
                continue
            if not isinstance(trace_value, Mapping):
                migrated_values[key] = trace_value
                continue
            trace_value_payload = dict(trace_value)
            if trace_value_payload.get("kind") == "absent":
                if version == 1:
                    trace_value_payload.setdefault(
                        "failure_code", "legacy.unknown"
                    )
                elif "failure_code" not in trace_value_payload:
                    raise ValueError(
                        "schema v2 Absent values require failure_code"
                    )
            migrated_values[key] = trace_value_payload

        if version == 1:
            payload["schema_version"] = TRACE_SCHEMA_VERSION
        payload["values"] = migrated_values
        return payload

    @model_validator(mode="after")
    def _validate_facts_are_json_lossless(self) -> SerializedTrace:
        reject_nonfinite_floats(self.step_facts, path="step_facts")
        return self


def serialize_trace(trace: Trace) -> SerializedTrace:
    """Total for traces built from TraceValue types. May lose warm
    caches, never information; round-trip must be value-equal (S3).
    """
    return SerializedTrace(
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
