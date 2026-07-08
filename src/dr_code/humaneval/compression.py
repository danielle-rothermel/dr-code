from __future__ import annotations

import gzip
from enum import StrEnum

import zstandard
from pydantic import BaseModel, ConfigDict


class CompressionMethod(StrEnum):
    GZIP = "gzip"
    ZSTD = "zstd"


class CompressionMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: CompressionMethod
    ground_truth_bytes: int
    representation_bytes: int
    compressed_bytes: int
    ratio_to_ground_truth: float | None
    percent_reduction_vs_ground_truth: float | None


CompressionMetrics = dict[CompressionMethod, CompressionMetric]
COMPRESSED_METRIC_METHODS = (CompressionMethod.GZIP, CompressionMethod.ZSTD)
ZSTD_COMPRESSOR = zstandard.ZstdCompressor()


def compressed_bytes(value: bytes, method: CompressionMethod) -> bytes:
    if method is CompressionMethod.GZIP:
        return gzip.compress(value)
    if method is CompressionMethod.ZSTD:
        return ZSTD_COMPRESSOR.compress(value)
    raise ValueError(f"unsupported compression method: {method}")


def compression_metrics(
    *,
    ground_truth_code: str,
    representation_text: str,
) -> CompressionMetrics:
    ground_truth_bytes = len(ground_truth_code.encode("utf-8"))
    representation = representation_text.encode("utf-8")
    representation_bytes = len(representation)
    metrics: CompressionMetrics = {}
    compressed_representations = {
        method: compressed_bytes(representation, method)
        for method in COMPRESSED_METRIC_METHODS
    }
    for method, compressed in compressed_representations.items():
        size = len(compressed)
        ratio = size / ground_truth_bytes if ground_truth_bytes else None
        metrics[method] = CompressionMetric(
            method=method,
            ground_truth_bytes=ground_truth_bytes,
            representation_bytes=representation_bytes,
            compressed_bytes=size,
            ratio_to_ground_truth=ratio,
            percent_reduction_vs_ground_truth=(
                (1.0 - ratio) * 100.0 if ratio is not None else None
            ),
        )
    return metrics
