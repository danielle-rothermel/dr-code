"""Authenticated sampled-observation identity."""

from __future__ import annotations

from typing import Self, cast

from dr_serialize import (
    Jsonable,
    build_identity_document,
    identity_document_hash,
)
from pydantic import StrictInt, field_validator, model_validator

from dr_code.models import FrozenModel

_SCHEMA_SAMPLE_IDENTITY = "dr_code.sample_identity"


def sample_identity_hash(
    *,
    sampling_config_identity: str,
    repeat_identity: str,
    ordinal: int,
    task_identity: str,
) -> str:
    document = build_identity_document(
        schema=_SCHEMA_SAMPLE_IDENTITY,
        schema_version=1,
        payload=cast(
            Jsonable,
            {
                "sampling_config_identity": sampling_config_identity,
                "repeat_identity": repeat_identity,
                "ordinal": ordinal,
                "task_identity": task_identity,
            },
        ),
    )
    return identity_document_hash(document)


class SampleIdentity(FrozenModel):
    """Authenticated identity for one sampled observation."""

    sampling_config_identity: str
    repeat_identity: str
    ordinal: StrictInt
    task_identity: str
    identity_hash: str

    @field_validator(
        "sampling_config_identity",
        "repeat_identity",
        "identity_hash",
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(
                "sample identity hashes must be lowercase SHA-256"
            )
        return value

    @field_validator("ordinal")
    @classmethod
    def validate_ordinal(cls, value: int) -> int:
        if value < 0:
            raise ValueError("sample ordinal must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        expected = sample_identity_hash(
            sampling_config_identity=self.sampling_config_identity,
            repeat_identity=self.repeat_identity,
            ordinal=self.ordinal,
            task_identity=self.task_identity,
        )
        if self.identity_hash != expected:
            raise ValueError("sample identity hash mismatch")
        return self


__all__ = ["SampleIdentity", "sample_identity_hash"]
