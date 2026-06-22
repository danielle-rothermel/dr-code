"""Compression metrics for decoder inputs (stage 4)."""

from __future__ import annotations

import zstandard

_ZSTD_LEVEL = 22
_COMPRESSOR = zstandard.ZstdCompressor(level=_ZSTD_LEVEL)


def decoder_input_compression(text: str) -> tuple[int, int]:
    """Return UTF-8 byte length and zstd22 compressed byte length."""
    raw_bytes = text.encode("utf-8")
    compressed = _COMPRESSOR.compress(raw_bytes)
    return len(raw_bytes), len(compressed)
