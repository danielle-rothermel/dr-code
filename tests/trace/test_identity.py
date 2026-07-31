"""Golden identity tests for trace boundary models."""

from __future__ import annotations

from dr_code.trace import (
    EXTERNAL_PRODUCER,
    EXTERNAL_PRODUCER_ID,
    JsonArtifact,
    TraceProducer,
    stable_hash,
)

JSON_ARTIFACT_GOLDEN_HASH = (
    "266924ee857ca4d0ca3dd88dfcc8d4f0ac837a8787d41aaaf2e16710c562d2a8"
    "12a7f7c51d18b49afaec0f0e2f4d15be5345fb225d269e8daac4b0bbe6c16cdc"
)


def test_external_producer_has_canonical_identity() -> None:
    assert EXTERNAL_PRODUCER_ID == "external"
    assert EXTERNAL_PRODUCER == TraceProducer(producer_id="external")
    assert EXTERNAL_PRODUCER.version is None
    assert EXTERNAL_PRODUCER.definition_hash is None


def test_stable_hash_matches_golden_identity() -> None:
    artifact = JsonArtifact(payload={"a": 1, "b": 2})

    assert stable_hash(artifact) == JSON_ARTIFACT_GOLDEN_HASH


def test_stable_hash_ignores_json_object_key_order() -> None:
    first = JsonArtifact(payload={"a": 1, "b": 2})
    reordered = JsonArtifact(payload={"b": 2, "a": 1})

    assert stable_hash(first) == stable_hash(reordered)
    assert stable_hash(reordered) == JSON_ARTIFACT_GOLDEN_HASH
