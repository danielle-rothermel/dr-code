"""Persisted trace migration and archive boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dr_code.trace import (
    ArchivedSerializedTraceV2,
    SerializedTrace,
    TextArtifact,
    Trace,
    TraceProducer,
    load_serialized_trace,
    serialize_trace,
)

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "persisted"


def _v3_payload() -> dict[str, object]:
    trace = Trace(
        values={
            "input": TextArtifact(text="input"),
            "output": TextArtifact(text="output"),
        },
        producer=TraceProducer(
            producer_id="preprocessing",
            version="1",
            definition_hash="a" * 64,
            preprocessing_config_hash="b" * 64,
            implementation_hash="c" * 64,
        ),
        step_facts={"parse": {"reason": "accepted"}},
    )
    payload = serialize_trace(trace).model_dump(mode="json")
    payload["schema_version"] = 3
    return payload


def test_trace_v3_migrates_losslessly_to_v4() -> None:
    payload = _v3_payload()

    loaded = load_serialized_trace(payload)

    assert isinstance(loaded, SerializedTrace)
    assert loaded.schema_version == 4
    expected = dict(payload)
    expected["schema_version"] = 4
    assert loaded.model_dump(mode="json") == expected


def test_trace_v3_migration_rejects_widened_step_facts() -> None:
    payload = _v3_payload()
    payload["step_facts"] = {"parse": {"count": 2}}

    with pytest.raises(ValidationError, match="string"):
        load_serialized_trace(payload)


def test_current_trace_model_rejects_direct_old_schema_validation() -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        SerializedTrace.model_validate(_v3_payload())


def test_published_trace_v2_fixture_is_readable_but_not_promotable() -> None:
    # Complete archive shape pinned from published commit 16286fdb.
    payload = (_FIXTURES / "trace_v2_16286fdb.json").read_text()

    loaded = load_serialized_trace(payload)

    assert isinstance(loaded, ArchivedSerializedTraceV2)
    assert loaded.model_dump(mode="json") == json.loads(payload)
    with pytest.raises(ValueError, match="cannot be promoted.*identity"):
        loaded.to_current()


def test_trace_loader_rejects_malformed_and_unknown_archives() -> None:
    historical = json.loads((_FIXTURES / "trace_v2_16286fdb.json").read_text())
    del historical["producer"]["preprocessing_config_hash"]
    with pytest.raises(ValidationError, match="preprocessing config hash"):
        load_serialized_trace(historical)

    with pytest.raises(ValueError, match="unsupported.*99"):
        load_serialized_trace({"schema_version": 99})

    with pytest.raises(ValueError, match="JSON object"):
        load_serialized_trace("[]")
