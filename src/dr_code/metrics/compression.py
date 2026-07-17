"""Compression primitives used by the compressed-length operator."""

from __future__ import annotations

import gzip
from enum import StrEnum
from typing import Annotated, Literal, Self

import zstandard
from pydantic import Field, model_validator

from dr_code.models import FrozenModel


class CompressionMethod(StrEnum):
    GZIP = "gzip"
    ZSTD = "zstd"


class GzipConfig(FrozenModel):
    method: Literal["gzip"] = "gzip"
    level: int

    @model_validator(mode="after")
    def validate_level(self) -> Self:
        if not 0 <= self.level <= 9:
            raise ValueError("gzip level must be between 0 and 9")
        return self


class ZstdConfig(FrozenModel):
    method: Literal["zstd"] = "zstd"
    level: int

    @model_validator(mode="after")
    def validate_level(self) -> Self:
        if self.level == 0 or self.level > 22:
            raise ValueError(
                "zstd level must be negative or between 1 and 22"
            )
        return self


CompressionConfig = Annotated[
    GzipConfig | ZstdConfig, Field(discriminator="method")
]


def compressed_bytes(
    value: bytes,
    *,
    method: CompressionMethod,
    level: int,
) -> bytes:
    """Compress ``value`` with an explicitly pinned codec level."""

    if method is CompressionMethod.GZIP:
        return gzip.compress(value, compresslevel=level)
    if method is CompressionMethod.ZSTD:
        return zstandard.ZstdCompressor(level=level).compress(value)
    raise ValueError(f"unsupported compression method: {method}")
