"""One explicitly parameterized compressed-length measurement."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self

from pydantic import model_validator

from dr_code.metrics.compression import CompressionConfig, compressed_bytes
from dr_code.metrics.names import MetricName
from dr_code.metrics.operators.base import (
    EngineContext,
    MetricOperator,
    OperatorResult,
    OperatorSettings,
    artifact_text,
)
from dr_code.trace import Artifact, ArtifactKind, CodeArtifact


class CompressedLengthSettings(OperatorSettings):
    compression: CompressionConfig
    reference_key: str | None = None

    @model_validator(mode="after")
    def validate_reference_key(self) -> Self:
        if self.reference_key == "":
            raise ValueError("reference_key must not be empty")
        return self


class CompressedLengthResult(OperatorResult):
    compressed_bytes: int
    representation_bytes: int


class CompressedLengthWithReferenceResult(CompressedLengthResult):
    ratio_to_reference: float | None
    percent_reduction: float | None


class CompressedLength(MetricOperator[CompressedLengthSettings]):
    NAME = MetricName.COMPRESSED_LENGTH
    VERSION = "0"
    INPUT = ArtifactKind.TEXT
    ACCEPTED_INPUTS = frozenset({ArtifactKind.TEXT, ArtifactKind.CODE})
    Settings = CompressedLengthSettings

    def auxiliary_keys(self) -> tuple[str, ...]:
        if self.settings.reference_key is None:
            return ()
        return (self.settings.reference_key,)

    def accepted_auxiliary_kinds(
        self,
        key: str,
    ) -> frozenset[ArtifactKind]:
        _ = key
        return frozenset({ArtifactKind.CODE})

    def compute(
        self,
        value: Artifact,
        aux: Mapping[str, Artifact],
        ctx: EngineContext,
    ) -> CompressedLengthResult:
        _ = ctx
        compression = self.settings.compression
        representation = artifact_text(value).encode("utf-8")
        size = len(
            compressed_bytes(
                representation,
                method=compression.method,
                level=compression.level,
            )
        )
        if self.settings.reference_key is None:
            return CompressedLengthResult(
                compressed_bytes=size,
                representation_bytes=len(representation),
            )

        reference = aux[self.settings.reference_key]
        if not isinstance(reference, CodeArtifact):
            raise TypeError("compression reference must be code")
        reference_bytes = len(reference.source.encode("utf-8"))
        ratio = size / reference_bytes if reference_bytes else None
        return CompressedLengthWithReferenceResult(
            compressed_bytes=size,
            representation_bytes=len(representation),
            ratio_to_reference=ratio,
            percent_reduction=(
                (1.0 - ratio) * 100.0 if ratio is not None else None
            ),
        )
