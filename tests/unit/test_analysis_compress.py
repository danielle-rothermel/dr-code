"""Unit tests for decoder_input compression metrics."""

from __future__ import annotations

from dr_code.analysis.compress import decoder_input_compression


def test_decoder_input_compression_returns_raw_and_zstd22_lengths() -> None:
    raw_len, zstd_len = decoder_input_compression("hello world")
    assert raw_len == 11
    assert zstd_len == 20
