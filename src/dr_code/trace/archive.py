"""Durable readers for current and published trace payloads."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal, NoReturn, Self, TypeAlias

from pydantic import model_validator

from dr_code.models import FrozenModel
from dr_code.trace.absent import Absent
from dr_code.trace.artifacts import Artifact
from dr_code.trace.provenance import EXTERNAL_PRODUCER_ID, TraceProducer
from dr_code.trace.serialization import SerializedTrace
from dr_code.trace.observation import SampleIdentity

PersistedTracePayload: TypeAlias = (
    Mapping[str, object] | str | bytes | bytearray
)


class ArchivedTraceProducerV2(FrozenModel):
    """Producer coordinates published before implementation/source identity."""

    producer_id: str
    version: str | None = None
    definition_hash: str | None = None
    preprocessing_config_hash: str | None = None

    @model_validator(mode="after")
    def validate_coordinates(self) -> Self:
        if not self.producer_id:
            raise ValueError("trace producer id must not be empty")
        if self.producer_id == EXTERNAL_PRODUCER_ID:
            if (
                self.version is not None
                or self.definition_hash is not None
                or self.preprocessing_config_hash is not None
            ):
                raise ValueError(
                    "external trace producers cannot claim definition "
                    "coordinates"
                )
            return self
        if not self.version:
            raise ValueError(
                "non-external trace producers require an explicit version"
            )
        _validate_sha256(self.definition_hash, owner="definition hash")
        _validate_sha256(
            self.preprocessing_config_hash,
            owner="preprocessing config hash",
        )
        return self


class ArchivedAbsentV2(FrozenModel):
    """Published absence shape from trace schema 2."""

    kind: Literal["absent"] = "absent"
    failed_step: str
    cause: str
    propagated_through: tuple[str, ...] = ()


ArchivedTraceValueV2: TypeAlias = Artifact | ArchivedAbsentV2


class ArchivedSerializedTraceV2(FrozenModel):
    """Readable trace schema 2, intentionally not promotable."""

    schema_version: Literal[2]
    producer: ArchivedTraceProducerV2
    values: dict[str, ArchivedTraceValueV2]
    step_facts: dict[str, dict[str, str]] = {}

    def to_current(self) -> NoReturn:
        """Fail closed instead of inventing authenticated identity evidence."""

        raise ValueError(
            "trace schema 2 cannot be promoted: it lacks authenticated "
            "producer implementation, sample identity, and external source "
            "evidence"
        )


class _SerializedTraceV3(FrozenModel):
    """Migration-only PR66 trace shape."""

    schema_version: Literal[3]
    producer: TraceProducer
    sample_identity: SampleIdentity | None = None
    values: dict[str, Artifact | Absent]
    step_facts: dict[str, dict[str, str]] = {}


LoadedSerializedTrace: TypeAlias = SerializedTrace | ArchivedSerializedTraceV2


def load_serialized_trace(
    payload: PersistedTracePayload,
) -> LoadedSerializedTrace:
    """Read one persisted trace, migrating only authenticated schema 3."""

    data = _payload_mapping(payload)
    schema_version = data.get("schema_version")
    if schema_version == 4:
        return SerializedTrace.model_validate(data)
    if schema_version == 3:
        previous = _SerializedTraceV3.model_validate(data)
        migrated = previous.model_dump(mode="python")
        migrated["schema_version"] = 4
        return SerializedTrace.model_validate(migrated)
    if schema_version == 2:
        return ArchivedSerializedTraceV2.model_validate(data)
    raise ValueError(
        f"unsupported serialized trace schema version: {schema_version!r}"
    )


def _payload_mapping(payload: PersistedTracePayload) -> dict[str, object]:
    if isinstance(payload, (str, bytes, bytearray)):
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise ValueError("serialized trace payload must be a JSON object")
        return decoded
    return dict(payload)


def _validate_sha256(value: str | None, *, owner: str) -> None:
    if (
        value is None
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{owner} must be a lowercase SHA-256")


__all__ = [
    "ArchivedAbsentV2",
    "ArchivedSerializedTraceV2",
    "ArchivedTraceProducerV2",
    "LoadedSerializedTrace",
    "load_serialized_trace",
]
