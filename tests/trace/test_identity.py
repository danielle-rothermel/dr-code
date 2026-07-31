"""Acceptance tests for stable_hash and provenance."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_code.trace import (
    EXTERNAL_PRODUCER_ID,
    CodeArtifact,
    ExternalSource,
    JsonArtifact,
    TextArtifact,
    TraceProducer,
    stable_hash,
)

# --- provenance ------------------------------------------------------


def test_external_producer_requires_caller_owned_identity() -> None:
    source = ExternalSource(source_id="fixture-a", content_digest="a" * 64)
    producer = TraceProducer(
        producer_id=EXTERNAL_PRODUCER_ID,
        external_source=source,
    )
    assert EXTERNAL_PRODUCER_ID == "external"
    assert producer.external_source == source
    assert producer.version is None
    assert producer.definition_hash is None


def test_non_external_trace_producer_requires_full_identity() -> None:
    with pytest.raises(ValidationError, match="explicit version"):
        TraceProducer(producer_id="preproc-1")


# --- stable_hash -----------------------------------------------------


def test_stable_hash_returns_str() -> None:
    assert isinstance(stable_hash(TextArtifact(text="hi")), str)


def test_stable_hash_deterministic_across_calls() -> None:
    art = CodeArtifact(source="x = 1\n")
    assert stable_hash(art) == stable_hash(art)


def test_stable_hash_key_order_insensitive() -> None:
    # sort_keys makes the hash field-order-proof: two JSON payloads that
    # differ only in key order must hash equal.
    a = JsonArtifact(payload={"a": 1, "b": 2})
    b = JsonArtifact(payload={"b": 2, "a": 1})
    assert stable_hash(a) == stable_hash(b)


def test_stable_hash_differs_for_different_inputs() -> None:
    a = TextArtifact(text="one")
    b = TextArtifact(text="two")
    assert stable_hash(a) != stable_hash(b)
