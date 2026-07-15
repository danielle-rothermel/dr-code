"""Acceptance tests for stable_hash and provenance."""

from __future__ import annotations

from dr_code.trace import (
    EXTERNAL_PRODUCER,
    EXTERNAL_PRODUCER_ID,
    CodeArtifact,
    JsonArtifact,
    TextArtifact,
    TraceProducer,
    stable_hash,
)

# --- provenance ------------------------------------------------------


def test_external_producer_id_constant() -> None:
    assert EXTERNAL_PRODUCER_ID == "external"


def test_external_producer_uses_external_id() -> None:
    # EXTERNAL_PRODUCER = TraceProducer(producer_id=EXTERNAL_PRODUCER_ID).
    assert EXTERNAL_PRODUCER.producer_id == EXTERNAL_PRODUCER_ID
    assert isinstance(EXTERNAL_PRODUCER, TraceProducer)


def test_trace_producer_optional_fields_default_none() -> None:
    producer = TraceProducer(producer_id="preproc-1")
    assert producer.version is None
    assert producer.definition_hash is None


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
