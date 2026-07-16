"""Compression primitives used by the compressed-length operator."""

from __future__ import annotations

import gzip
from enum import StrEnum

import zstandard


class CompressionMethod(StrEnum):
    GZIP = "gzip"
    ZSTD = "zstd"


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
