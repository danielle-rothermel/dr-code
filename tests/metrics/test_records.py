"""Vocabulary and record contracts (plan sections ``names.py`` / ``records.py``).

Covers ``MetricName`` / ``RecordStatus`` (StrEnum members), registry↔enum
sync, ``MetricRecord`` (identity + lineage + exactly-one-shape-per-status),
and ``record_rows`` flattening with ``"{metric}.{key}"`` value columns
(design X-S1, X-S3, L2/L3).

``dr_code.metrics`` is imported lazily inside each test so the suite collects
cleanly against the missing package and fails hard (never skips) when absent.
"""

from __future__ import annotations

import pytest

EXPECTED_METRIC_NAMES = {
    "text_stats",
    "code_leakage",
    "parse_outcome",
    "ast_stats",
    "compressed_length",
    "code_test",
}
EXPECTED_RECORD_STATUSES = {
    "measured",
    "not_applicable",
    "operator_failure",
}


# ===========================================================================
# MetricName / RecordStatus enums.
# ===========================================================================

def test_metric_name_is_a_strenum_of_the_six_families() -> None:
    from dr_code.metrics import MetricName

    assert {name.value for name in MetricName} == EXPECTED_METRIC_NAMES


def test_metric_name_members_round_trip_through_their_string_values() -> None:
    from dr_code.metrics import MetricName

    for value in EXPECTED_METRIC_NAMES:
        name = MetricName(value)
        assert name.value == value
        assert name == str(name)  # StrEnum serializes to plain JSON


def test_record_status_is_the_three_way_answer_taxonomy() -> None:
    from dr_code.metrics import RecordStatus

    assert {status.value for status in RecordStatus} == EXPECTED_RECORD_STATUSES


def test_record_status_members_round_trip_through_their_string_values() -> None:
    from dr_code.metrics import RecordStatus

    for value in EXPECTED_RECORD_STATUSES:
        status = RecordStatus(value)
        assert status.value == value


# ---------------------------------------------------------------------------
# Registry ↔ enum sync (plan: ``names.py`` registry↔enum sync-tested).
# ---------------------------------------------------------------------------

def test_registry_covers_every_metric_name() -> None:
    from dr_code.metrics import MetricName
    from dr_code.metrics.registry import REGISTRY

    for name in MetricName:
        assert str(name) in REGISTRY, f"{name} missing from REGISTRY"


def test_registry_has_no_stray_keys() -> None:
    from dr_code.metrics import MetricName
    from dr_code.metrics.registry import REGISTRY

    enum_values = {str(name) for name in MetricName}
    for key in REGISTRY:
        assert key in enum_values, f"registry key {key!r} not in MetricName"


# ===========================================================================
# MetricRecord schema + identity/lineage.
# ===========================================================================

def _identity_kwargs(**overrides: object) -> dict[str, object]:
    base = dict(
        metric=None,
        metric_version="1",
        on_key="input",
        producer_id="pre",
        producer_version="v1",
        producer_definition_hash="abc",
        metrics_definition_id="def",
        metrics_definition_version="1",
    )
    base.update(overrides)
    return base


def _record(**overrides: object):
    from dr_code.metrics import MetricName, MetricRecord, RecordStatus

    kw = _identity_kwargs()
    kw["metric"] = MetricName.TEXT_STATS
    kw["status"] = RecordStatus.MEASURED
    kw["values"] = {"character_count": 4}
    kw.update(overrides)
    return MetricRecord(**kw)


def test_metric_record_field_set_is_the_documented_schema() -> None:
    from dr_code.metrics.records import MetricRecord

    assert set(MetricRecord.model_fields) == {
        "metric",
        "metric_version",
        "settings",
        "on_key",
        "producer_id",
        "producer_version",
        "producer_definition_hash",
        "metrics_definition_id",
        "metrics_definition_version",
        "status",
        "values",
        "absence_failed_step",
        "absence_cause",
        "failure_type",
        "failure_message",
    }


def test_metric_record_is_frozen() -> None:
    record = _record()
    with pytest.raises(Exception):  # noqa: PT011 — FrozenModel raises
        record.metric = record.metric  # type: ignore[misc]


def test_metric_record_defaults_settings_to_empty() -> None:
    assert _record().settings == {}


def test_equal_records_compare_equal() -> None:
    """Records participate in equality-based comparison across runs (X-M1).
    Deterministic content identity is metrics_definition_hash / record_rows,
    not Python __hash__ (records carry dict values)."""
    assert _record() == _record()


def test_metric_record_carries_identity_and_lineage() -> None:
    record = _record()
    assert record.metric_version == "1"
    assert record.settings == {}
    assert record.on_key == "input"
    assert record.producer_id == "pre"
    assert record.producer_version == "v1"
    assert record.producer_definition_hash == "abc"
    assert record.metrics_definition_id == "def"
    assert record.metrics_definition_version == "1"


def test_metric_record_values_accept_all_scalar_types() -> None:
    """MetricScalar = float | int | str | bool | None."""
    record = _record(
        values={
            "int_val": 42,
            "float_val": 3.14,
            "str_val": "hello",
            "bool_val": True,
            "none_val": None,
        },
    )
    assert record.values["int_val"] == 42
    assert record.values["float_val"] == 3.14
    assert record.values["str_val"] == "hello"
    assert record.values["bool_val"] is True
    assert record.values["none_val"] is None


# ===========================================================================
# Exactly-one-shape-per-status (X-S1).
# ===========================================================================

def test_measured_record_shape() -> None:
    from dr_code.metrics import RecordStatus

    record = _record(values={"character_count": 4})
    assert record.status is RecordStatus.MEASURED
    assert record.values == {"character_count": 4}
    assert record.absence_failed_step is None
    assert record.absence_cause is None
    assert record.failure_type is None
    assert record.failure_message is None


def test_not_applicable_record_carries_causal_lineage() -> None:
    from dr_code.metrics import RecordStatus

    record = _record(
        status=RecordStatus.NOT_APPLICABLE,
        values={},
        absence_failed_step="extract",
        absence_cause="no code extracted",
    )
    assert record.status is RecordStatus.NOT_APPLICABLE
    assert record.values == {}
    assert record.absence_failed_step == "extract"
    assert record.absence_cause == "no code extracted"


def test_operator_failure_record_is_attributed_to_the_metric() -> None:
    from dr_code.metrics import RecordStatus

    record = _record(
        status=RecordStatus.OPERATOR_FAILURE,
        values={},
        failure_type="ValueError",
        failure_message="boom",
    )
    assert record.status is RecordStatus.OPERATOR_FAILURE
    assert record.values == {}
    assert record.failure_type == "ValueError"
    assert record.failure_message == "boom"


def test_measured_record_rejects_absence_and_failure_fields() -> None:
    """The three-way shape contract (X-S1): MEASURED carries values only."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _record(absence_failed_step="parse")
    with pytest.raises(ValidationError):
        _record(failure_type="ValueError")


def test_measured_record_rejects_empty_values() -> None:
    """An empty values dict on a MEASURED record is indistinguishable from the
    no-answer shape — the model rejects it."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _record(values={})


def test_not_applicable_record_requires_failed_step_and_cause() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import RecordStatus

    base = dict(
        status=RecordStatus.NOT_APPLICABLE,
        values={},
        absence_failed_step="extract",
        absence_cause="no code",
    )
    for removed in ("absence_failed_step", "absence_cause"):
        kw = {**base}
        kw.pop(removed)
        with pytest.raises(ValidationError):
            _record(**kw)


def test_operator_failure_record_requires_type_and_message() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import RecordStatus

    base = dict(
        status=RecordStatus.OPERATOR_FAILURE,
        values={},
        failure_type="ValueError",
        failure_message="boom",
    )
    for removed in ("failure_type", "failure_message"):
        kw = {**base}
        kw.pop(removed)
        with pytest.raises(ValidationError):
            _record(**kw)


def test_records_differ_on_metric_settings_on_key_and_status() -> None:
    from dr_code.metrics import MetricName, RecordStatus

    a = _record()
    assert a != _record(metric=MetricName.AST_STATS)
    assert a != _record(
        metric=MetricName.COMPRESSED_LENGTH,
        settings={"method": "gzip", "level": 9},
    )
    assert a != _record(on_key="output")
    measured = _record()
    absent = _record(
        status=RecordStatus.NOT_APPLICABLE,
        values={},
        absence_failed_step="x",
        absence_cause="y",
    )
    assert measured != absent


# ===========================================================================
# record_rows: flat dataframe rows, metric-prefixed value columns (X-S3).
# ===========================================================================

def test_record_rows_returns_one_row_per_record() -> None:
    from dr_code.metrics import MetricName, record_rows

    rows = record_rows([_record(), _record(metric=MetricName.AST_STATS)])
    assert len(rows) == 2
    assert all(isinstance(row, dict) for row in rows)


def test_record_rows_empty_input_returns_empty_list() -> None:
    from dr_code.metrics import record_rows

    assert record_rows([]) == []


def test_record_rows_prefix_value_columns_with_metric_and_key() -> None:
    from dr_code.metrics import record_rows

    rows = record_rows(
        [_record(values={"character_count": 4, "word_count": 1})]
    )
    row = rows[0]
    assert row["text_stats.character_count"] == 4
    assert row["text_stats.word_count"] == 1
    # Raw value keys never appear un-prefixed (collision avoidance, X-S3).
    assert "character_count" not in row
    assert "word_count" not in row


def test_record_rows_include_identity_and_lineage_columns() -> None:
    from dr_code.metrics import MetricName, RecordStatus, record_rows

    row = record_rows([_record()])[0]
    assert row["metric"] == MetricName.TEXT_STATS
    assert row["metric_version"] == "1"
    assert row["on_key"] == "input"
    assert row["producer_id"] == "pre"
    assert row["metrics_definition_id"] == "def"
    assert row["status"] == RecordStatus.MEASURED
    assert row["settings"] == {}


def test_record_rows_value_columns_are_collision_free_across_metrics() -> None:
    from dr_code.metrics import MetricName, record_rows

    rows = record_rows(
        [
            _record(metric=MetricName.TEXT_STATS, values={"count": 1}),
            _record(metric=MetricName.AST_STATS, values={"count": 2}),
        ]
    )
    assert rows[0]["text_stats.count"] == 1
    assert rows[1]["ast_stats.count"] == 2


def test_record_rows_status_column_distinguishes_absence_from_measured_zero() -> None:
    """Not-applicable ≠ measured zero: a status column, not a magic value."""
    from dr_code.metrics import RecordStatus, record_rows

    measured_zero = _record(values={"count": 0})
    not_applicable = _record(
        status=RecordStatus.NOT_APPLICABLE,
        values={},
        absence_failed_step="extract",
        absence_cause="absent",
    )
    rows = record_rows([measured_zero, not_applicable])
    assert rows[0]["status"] == RecordStatus.MEASURED
    assert rows[0]["text_stats.count"] == 0
    assert rows[1]["status"] == RecordStatus.NOT_APPLICABLE
    assert "text_stats.count" not in rows[1]
    assert rows[1]["absence_failed_step"] == "extract"
    assert rows[1]["absence_cause"] == "absent"


def test_record_rows_preserve_declaration_order() -> None:
    from dr_code.metrics import MetricName, record_rows

    rows = record_rows(
        [
            _record(metric=MetricName.TEXT_STATS),
            _record(metric=MetricName.CODE_LEAKAGE),
            _record(metric=MetricName.AST_STATS),
        ]
    )
    assert [row["metric"] for row in rows] == [
        MetricName.TEXT_STATS,
        MetricName.CODE_LEAKAGE,
        MetricName.AST_STATS,
    ]
