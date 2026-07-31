"""Authenticated trace producer contracts."""

from __future__ import annotations

from typing import Final

from pydantic import model_validator
from typing import Self

from dr_code.models import FrozenModel

EXTERNAL_PRODUCER_ID: Final = "external"


class ExternalSource(FrozenModel):
    """Caller-owned identity and content digest for an external source."""

    source_id: str
    content_digest: str

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if not self.source_id:
            raise ValueError("external source id must not be empty")
        if len(self.content_digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.content_digest
        ):
            raise ValueError(
                "external source content digest must be a lowercase SHA-256"
            )
        return self


class TraceProducer(FrozenModel):
    """Identifies the producing definition and concrete config."""

    # preprocessing definition_id, or "external"
    producer_id: str
    version: str | None = None
    # stable_hash of the full frozen definition
    definition_hash: str | None = None
    preprocessing_config_hash: str | None = None
    implementation_hash: str | None = None
    external_source: ExternalSource | None = None

    @model_validator(mode="after")
    def validate_coordinates(self) -> Self:
        if not self.producer_id:
            raise ValueError("trace producer id must not be empty")
        if self.producer_id == EXTERNAL_PRODUCER_ID:
            if (
                self.version is not None
                or self.definition_hash is not None
                or self.preprocessing_config_hash is not None
                or self.implementation_hash is not None
            ):
                raise ValueError(
                    "external trace producers cannot claim definition coordinates"
                )
            if self.external_source is None:
                raise ValueError(
                    "external trace producers require a caller-supplied source"
                )
            return self
        if self.external_source is not None:
            raise ValueError(
                "preprocessing trace producers cannot claim an external source"
            )
        if not self.version:
            raise ValueError(
                "non-external trace producers require an explicit version"
            )
        if (
            self.definition_hash is None
            or len(self.definition_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.definition_hash
            )
        ):
            raise ValueError(
                "non-external trace producers require a lowercase SHA-256 "
                "definition hash"
            )
        if (
            self.preprocessing_config_hash is None
            or len(self.preprocessing_config_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.preprocessing_config_hash
            )
        ):
            raise ValueError(
                "non-external trace producers require a lowercase SHA-256 "
                "preprocessing config hash"
            )
        if (
            self.implementation_hash is None
            or len(self.implementation_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.implementation_hash
            )
        ):
            raise ValueError(
                "non-external trace producers require a lowercase SHA-256 "
                "implementation hash"
            )
        return self
