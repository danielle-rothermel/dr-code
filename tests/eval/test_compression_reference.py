"""Generic compression references and explicit zero-denominator behavior."""

from __future__ import annotations

import pytest

from dr_code.eval.compression_reference import (
    ZERO_DENOMINATOR,
    CompressionReferenceArtifact,
    CompressionReferenceKey,
    CompressionReferenceResolver,
    ReferenceResolutionError,
    compression_ratio,
)


def test_key_resolves_to_bound_artifact() -> None:
    key = CompressionReferenceKey(namespace="ns", name="ref")
    artifact = CompressionReferenceArtifact(content=b"reference-bytes")
    resolver = CompressionReferenceResolver.from_mapping({key: artifact})
    assert resolver.resolve(key) == artifact


def test_unbound_key_raises_explicitly() -> None:
    resolver = CompressionReferenceResolver()
    key = CompressionReferenceKey(namespace="ns", name="missing")
    with pytest.raises(ReferenceResolutionError):
        resolver.resolve(key)


def test_zero_denominator_is_explicit_not_coerced() -> None:
    empty = CompressionReferenceArtifact(content=b"")
    result = compression_ratio(numerator_bytes=10, reference=empty)
    # Explicitly the ZERO_DENOMINATOR sentinel, never coerced to 0.0/1.0.
    assert result is ZERO_DENOMINATOR
    assert result is None


def test_nonzero_denominator_divides() -> None:
    reference = CompressionReferenceArtifact(content=b"abcd")  # 4 bytes
    assert compression_ratio(numerator_bytes=2, reference=reference) == 0.5


def test_generic_layer_has_no_dataset_field_knowledge() -> None:
    # The key is a plain namespaced identifier; no field selection here.
    key = CompressionReferenceKey(namespace="dataset", name="gt_code")
    assert key.namespace == "dataset"
    assert key.name == "gt_code"
    assert len(key.identity_hash()) == 64


def test_artifact_identity_tracks_bytes() -> None:
    a = CompressionReferenceArtifact(content=b"abc")
    b = CompressionReferenceArtifact(content=b"abd")
    assert a.identity_hash() != b.identity_hash()
    assert a.byte_length == 3
