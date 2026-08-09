from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal, TypeAlias

from dr_serialize import Sha256Digest
from dr_store import ObjectReference
from pydantic import Field, PositiveInt, field_validator

from dr_code.core.models import FrozenModel


class BundleRecordReference(FrozenModel):
    kind: Literal["bundle_record"] = "bundle_record"
    artifact_name: str
    record_index: int = Field(ge=0)
    record_sha256: Sha256Digest
    schema: str
    schema_version: PositiveInt

    @field_validator("artifact_name")
    @classmethod
    def validate_artifact_name(cls, artifact_name: str) -> str:
        path = PurePosixPath(artifact_name)
        if (
            not artifact_name
            or artifact_name.startswith("/")
            or path.as_posix() != artifact_name
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError(
                "artifact_name must be a normalized relative artifact name"
            )
        return artifact_name


class StoredRecordReference(FrozenModel):
    kind: Literal["stored_record"] = "stored_record"
    reference: ObjectReference
    schema_version: PositiveInt


EvidenceReference: TypeAlias = Annotated[
    BundleRecordReference | StoredRecordReference,
    Field(discriminator="kind"),
]


__all__ = [
    "BundleRecordReference",
    "EvidenceReference",
    "StoredRecordReference",
]
