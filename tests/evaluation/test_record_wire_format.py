from __future__ import annotations

import json

from .test_record_models import attempt, evaluated, execution, reference
from dr_code.evaluation import (
    CANDIDATE_EXECUTION_RECORD_SCHEMA_VERSION,
    EVALUATION_ATTEMPT_SCHEMA_VERSION,
    SAMPLE_EVALUATION_RECORD_SCHEMA_VERSION,
)


def test_record_schema_versions_are_pinned() -> None:
    assert EVALUATION_ATTEMPT_SCHEMA_VERSION == 1
    assert SAMPLE_EVALUATION_RECORD_SCHEMA_VERSION == 1
    assert CANDIDATE_EXECUTION_RECORD_SCHEMA_VERSION == 1


def test_reference_wire_keys_and_discriminator_are_exact() -> None:
    payload = json.loads(reference().model_dump_json())
    assert list(payload) == [
        "kind",
        "artifact_name",
        "record_index",
        "record_sha256",
        "schema",
        "schema_version",
    ]
    assert payload["kind"] == "bundle_record"


def test_candidate_execution_wire_keys_and_discriminators_are_exact() -> None:
    payload = json.loads(execution().model_dump_json())
    assert list(payload) == [
        "schema_version",
        "candidate",
        "request_identity",
        "runtime",
        "cache_namespace",
        "cache_key",
        "provenance",
        "outcome",
    ]
    assert payload["provenance"]["kind"] == "reused"
    assert payload["outcome"]["kind"] == "harness_failure"


def test_sample_record_wire_has_one_raw_input_occurrence() -> None:
    payload = json.loads(evaluated().model_dump_json())
    assert list(payload) == [
        "schema_version",
        "status",
        "slot",
        "sample",
        "trace",
        "candidates",
        "executions",
        "metrics",
    ]
    assert payload["status"] == "evaluated"
    assert "raw_input" not in payload
    assert payload["trace"]["values"]["input"]["text"] == "raw input"


def test_attempt_wire_keys_and_labels_are_exact() -> None:
    payload = json.loads(attempt().model_dump_json())
    assert list(payload) == [
        "schema_version",
        "identity",
        "plan",
        "runtime",
        "cache_namespace",
        "members",
        "completeness",
        "validity",
        "limit_exhaustion",
        "replay",
    ]
    assert payload["completeness"] == "complete"
    assert payload["validity"] == "valid"
