"""Persisted MetricRecord migration and archive boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dr_code.eval import (
    AbsenceMode,
    Applicability,
    ArchivedMetricRecordV1,
    LEGACY_UNSPECIFIED_FAILURE_CODE,
    MetricFact,
    MetricRecord,
    OperatorCoordinates,
    OperatorLineage,
    RecordStatus,
    load_metric_record,
)
from dr_code.trace import TraceProducer

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "persisted"
_PROCEDURE_HASH = "d" * 64


def _producer() -> TraceProducer:
    return TraceProducer(
        producer_id="preprocessing",
        version="1",
        definition_hash="a" * 64,
        preprocessing_config_hash="b" * 64,
        implementation_hash="c" * 64,
    )


def _operator() -> OperatorCoordinates:
    return OperatorCoordinates(
        name="text_stats",
        version="1",
        implementation_hash="c" * 64,
        settings=(),
    )


def _v2_payload(status: RecordStatus) -> dict[str, object]:
    operator = _operator()
    common = {
        "question": operator.name,
        "question_identity_hash": operator.question_identity_hash(
            on_key="output"
        ),
        "on_key": "output",
        "evaluation_procedure_config_hash": _PROCEDURE_HASH,
        "trace_producer": _producer(),
        "operator": operator,
    }
    if status is RecordStatus.NOT_APPLICABLE:
        current = MetricRecord.not_applicable(
            **common,
            absence_mode=AbsenceMode.MISSING_TRACE_KEY,
            cause="output is absent",
            failure_code="current.failure",
        )
    elif status is RecordStatus.MEASURED:
        current = MetricRecord.measured(
            **common,
            facts=(
                MetricFact(
                    name="word_count",
                    value=2,
                    unit="word",
                    applicability=Applicability.APPLICABLE,
                    lineage=OperatorLineage(
                        evaluation_procedure_config_hash=_PROCEDURE_HASH,
                        question_identity_hash=common[
                            "question_identity_hash"
                        ],
                        operator=operator.name,
                        operator_version=operator.version,
                        operator_implementation=operator.implementation_hash,
                    ),
                ),
            ),
        )
    else:
        current = MetricRecord.operator_failure(
            **common,
            failure_type="ValueError",
            failure_message="operator failed",
        )
    payload = current.model_dump(mode="json")
    payload["schema_version"] = 2
    payload.pop("failure_code")
    return payload


def test_metric_record_v2_not_applicable_migrates_with_reserved_code() -> None:
    payload = _v2_payload(RecordStatus.NOT_APPLICABLE)

    loaded = load_metric_record(payload)

    assert isinstance(loaded, MetricRecord)
    assert loaded.schema_version == 3
    assert loaded.failure_code == LEGACY_UNSPECIFIED_FAILURE_CODE
    assert loaded.absence_mode is AbsenceMode.MISSING_TRACE_KEY
    assert loaded.absence_cause == "output is absent"


@pytest.mark.parametrize(
    "status",
    [RecordStatus.MEASURED, RecordStatus.OPERATOR_FAILURE],
)
def test_metric_record_v2_other_status_preserves_shape(
    status: RecordStatus,
) -> None:
    payload = _v2_payload(status)

    loaded = load_metric_record(json.dumps(payload))

    assert isinstance(loaded, MetricRecord)
    assert loaded.schema_version == 3
    assert loaded.status is status
    assert loaded.failure_code is None
    expected = dict(payload)
    expected["schema_version"] = 3
    expected["failure_code"] = None
    assert loaded.model_dump(mode="json") == expected


@pytest.mark.parametrize("schema_version", [1, 2])
def test_current_metric_record_rejects_direct_old_schema_validation(
    schema_version: int,
) -> None:
    payload = _v2_payload(RecordStatus.OPERATOR_FAILURE)
    payload["schema_version"] = schema_version
    with pytest.raises(ValidationError, match="schema_version"):
        MetricRecord.model_validate(payload)


def test_published_metric_v1_fixture_is_readable_but_not_promotable() -> None:
    # Complete archive shape pinned from published commit 16286fdb.
    payload = (_FIXTURES / "metric_record_v1_16286fdb.json").read_text()

    loaded = load_metric_record(payload)

    assert isinstance(loaded, ArchivedMetricRecordV1)
    assert loaded.model_dump(mode="json") == json.loads(payload)
    with pytest.raises(ValueError, match="cannot be promoted.*identity"):
        loaded.to_current()


def test_metric_record_loader_rejects_malformed_and_unknown_payloads() -> None:
    malformed = _v2_payload(RecordStatus.NOT_APPLICABLE)
    malformed["absence_cause"] = None
    with pytest.raises(ValidationError, match="absence mode, cause"):
        load_metric_record(malformed)

    with pytest.raises(ValueError, match="unsupported.*99"):
        load_metric_record({"schema_version": 99})

    with pytest.raises(ValueError, match="JSON object"):
        load_metric_record("[]")


def test_metric_record_v2_rejects_fields_not_owned_by_pr66() -> None:
    payload = _v2_payload(RecordStatus.NOT_APPLICABLE)
    payload["failure_code"] = "not-yet-in-schema"

    with pytest.raises(ValidationError, match="failure_code"):
        load_metric_record(payload)
