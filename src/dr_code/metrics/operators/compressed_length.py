"""One explicitly parameterized compressed-length measurement."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self

from pydantic import model_validator

from dr_code.metrics.compression import CompressionMethod, compressed_bytes
from dr_code.metrics.names import MetricName
from dr_code.metrics.operators.base import (
    EngineContext,
    MetricOperator,
    OperatorSettings,
    artifact_text,
)
from dr_code.metrics.records import MetricScalar
from dr_code.trace import Artifact, ArtifactKind, CodeArtifact


class CompressedLengthSettings(OperatorSettings):
    method: CompressionMethod
    level: int
    reference_key: str | None = None

    @model_validator(mode="after")
    def validate_level(self) -> Self:
        if self.method is CompressionMethod.GZIP:
            if not 0 <= self.level <= 9:
                raise ValueError("gzip level must be between 0 and 9")
        elif self.level == 0 or self.level > 22:
            raise ValueError(
                "zstd level must be negative or between 1 and 22"
            )
        if self.reference_key == "":
            raise ValueError("reference_key must not be empty")
        return self


class CompressedLength(MetricOperator):
    NAME = MetricName.COMPRESSED_LENGTH
    VERSION = "1"
    INPUT = ArtifactKind.TEXT
    ACCEPTED_INPUTS = frozenset({ArtifactKind.TEXT, ArtifactKind.CODE})
    Settings = CompressedLengthSettings

    def auxiliary_keys(self) -> tuple[str, ...]:
        settings = self.settings
        assert isinstance(settings, CompressedLengthSettings)
        if settings.reference_key is None:
            return ()
        return (settings.reference_key,)

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
    ) -> dict[str, MetricScalar]:
        _ = ctx
        settings = self.settings
        assert isinstance(settings, CompressedLengthSettings)
        representation = artifact_text(value).encode("utf-8")
        size = len(
            compressed_bytes(
                representation,
                method=settings.method,
                level=settings.level,
            )
        )
        values: dict[str, MetricScalar] = {
            "compressed_bytes": size,
            "representation_bytes": len(representation),
        }
        if settings.reference_key is None:
            return values

        reference = aux[settings.reference_key]
        if not isinstance(reference, CodeArtifact):
            raise TypeError("compression reference must be code")
        reference_bytes = len(reference.source.encode("utf-8"))
        ratio = size / reference_bytes if reference_bytes else None
        values.update(
            {
                "ratio_to_reference": ratio,
                "percent_reduction": (
                    (1.0 - ratio) * 100.0 if ratio is not None else None
                ),
            }
        )
        return values
