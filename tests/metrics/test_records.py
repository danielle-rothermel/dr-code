"""Metrics vocabulary and record contracts.

Covers ``MetricName`` / ``RecordStatus`` (StrEnum members), registry↔enum
sync, ``MetricRecord`` (identity + lineage + exactly-one-shape-per-status),
and ``record_rows`` flattening with ``"{metric}.{key}"`` value columns
that prevent collisions across metrics.
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

    assert {
        status.value for status in RecordStatus
    } == EXPECTED_RECORD_STATUSES


def test_record_status_members_round_trip_through_their_string_values() -> (
    None
):
    from dr_code.metrics import RecordStatus

    for value in EXPECTED_RECORD_STATUSES:
        status = RecordStatus(value)
        assert status.value == value


# ---------------------------------------------------------------------------
# Registry and enum synchronization.
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
    from dr_code.metrics import MetricName, MetricQuestion, MetricsDefinition
    from dr_code.trace import (
        ComponentCoordinate,
        PreprocessingDefinitionCoordinate,
        PreprocessingTraceProducer,
        StepCoordinate,
    )

    base = dict(
        metric=None,
        metric_version="1",
        on_key="input",
        producer=PreprocessingTraceProducer(
            definition=PreprocessingDefinitionCoordinate(
                definition_id="pre",
                version="v1",
                steps=(
                    StepCoordinate(
                        instance_name="step",
                        component=ComponentCoordinate(
                            registered_name="normalize_unicode",
                            version="0",
                        ),
                    ),
                ),
            )
        ),
        metrics_definition=MetricsDefinition(
            definition_id="def",
            version="1",
            questions=(
                MetricQuestion(metric=MetricName.TEXT_STATS, on="input"),
            ),
        ),
    )
    base.update(overrides)
    return base


def _matching_definition(
    metric: object, on_key: object, settings: object
) -> object:
    """The nested lineage a record's own identity must be declared in."""
    from dr_code.metrics import MetricQuestion, MetricsDefinition

    return MetricsDefinition(
        definition_id="def",
        version="1",
        questions=(
            MetricQuestion(metric=metric, on=on_key, settings=settings),
        ),
    )


def _record(**overrides: object):
    from dr_code.metrics import MetricName, MetricRecord, RecordStatus

    kw = _identity_kwargs()
    kw["metric"] = MetricName.TEXT_STATS
    kw["status"] = RecordStatus.MEASURED
    kw["values"] = {"character_count": 4}
    kw.update(overrides)
    if "metrics_definition" not in overrides:
        kw["metrics_definition"] = _matching_definition(
            kw["metric"], kw["on_key"], kw.get("settings", {})
        )
    return MetricRecord(**kw)


def test_metric_record_field_set_is_the_documented_schema() -> None:
    from dr_code.metrics.records import MetricRecord

    assert set(MetricRecord.model_fields) == {
        "metric",
        "metric_version",
        "settings",
        "on_key",
        "producer",
        "metrics_definition",
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
    from dr_code.metrics.operators.base import OperatorSettings

    assert _record().settings == OperatorSettings()


def test_equal_records_compare_equal() -> None:
    """Records participate in structured comparison across runs."""
    assert _record() == _record()


def test_metric_record_carries_identity_and_lineage() -> None:
    from dr_code.metrics.operators.base import OperatorSettings

    record = _record()
    assert record.metric_version == "1"
    assert record.settings == OperatorSettings()
    assert record.on_key == "input"
    assert record.producer.kind == "preprocessing"
    assert record.producer.definition.definition_id == "pre"
    assert record.producer.definition.version == "v1"
    assert record.metrics_definition.definition_id == "def"
    assert record.metrics_definition.version == "1"
    assert tuple(
        question.metric for question in record.metrics_definition.questions
    ) == (record.metric,)


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
# Exactly one field shape per status.
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
    """The three-way shape contract gives MEASURED records values only."""
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
        settings={"compression": {"method": "gzip", "level": 9}},
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
# record_rows: flat rows with metric-prefixed value columns.
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
    # Raw value keys never appear unprefixed, avoiding cross-metric collisions.
    assert "character_count" not in row
    assert "word_count" not in row


def test_record_rows_include_identity_and_lineage_columns() -> None:
    from dr_code.metrics import MetricName, RecordStatus, record_rows

    row = record_rows([_record()])[0]
    assert row["metric"] == MetricName.TEXT_STATS
    assert row["metric_version"] == "1"
    assert row["on_key"] == "input"
    assert row["producer"]["definition"]["definition_id"] == "pre"
    assert row["metrics_definition"]["definition_id"] == "def"
    assert row["metrics_definition"]["version"] == "1"
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


def test_record_rows_status_column_distinguishes_absence_from_measured_zero() -> (
    None
):
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


# ===========================================================================
# Settings belong to the named metric; the discriminator is required.
# ===========================================================================


def test_metric_question_rejects_settings_from_another_operator() -> None:
    """Another operator's settings model is revalidated, not waved through."""
    from pydantic import ValidationError

    from dr_code.metrics import MetricName, MetricQuestion
    from dr_code.metrics.operators.code_leakage import CodeLeakageSettings

    with pytest.raises(ValidationError):
        MetricQuestion(
            metric=MetricName.TEXT_STATS,
            on="output",
            settings=CodeLeakageSettings(task_names=("x",)),
        )


def test_metric_question_accepts_its_own_settings_instance_and_dict() -> None:
    from dr_code.metrics import MetricName, MetricQuestion
    from dr_code.metrics.operators.code_leakage import CodeLeakageSettings

    expected = CodeLeakageSettings(task_names=("x",))
    from_instance = MetricQuestion(
        metric=MetricName.CODE_LEAKAGE, on="output", settings=expected
    )
    from_dict = MetricQuestion(
        metric=MetricName.CODE_LEAKAGE,
        on="output",
        settings={"task_names": ["x"]},
    )
    assert from_instance.settings == expected
    assert from_dict.settings == expected


def test_metric_question_missing_metric_raises_validation_error() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import MetricQuestion

    with pytest.raises(ValidationError):
        MetricQuestion.model_validate({"on": "output", "settings": {}})


def test_metric_record_rejects_settings_from_another_operator() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import MetricName
    from dr_code.metrics.operators.code_leakage import CodeLeakageSettings

    with pytest.raises(ValidationError):
        _record(
            metric=MetricName.TEXT_STATS,
            settings=CodeLeakageSettings(task_names=("x",)),
        )


def test_metric_record_accepts_its_own_settings_instance_and_dict() -> None:
    from dr_code.metrics import MetricName
    from dr_code.metrics.operators.code_leakage import CodeLeakageSettings

    expected = CodeLeakageSettings(task_names=("x",))
    from_instance = _record(metric=MetricName.CODE_LEAKAGE, settings=expected)
    from_dict = _record(
        metric=MetricName.CODE_LEAKAGE, settings={"task_names": ["x"]}
    )
    assert from_instance.settings == expected
    assert from_dict.settings == expected


def test_metric_record_missing_metric_raises_validation_error() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import MetricRecord

    with pytest.raises(ValidationError):
        MetricRecord.model_validate(
            {"metric_version": "0", "settings": {}, "on_key": "o"}
        )


# ===========================================================================
# A record answers a question its own nested definition declares.
# ===========================================================================


def test_record_identity_must_name_a_question_it_nests() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import (
        MetricName,
        MetricQuestion,
        MetricsDefinition,
    )

    elsewhere = MetricsDefinition(
        definition_id="def",
        version="1",
        questions=(MetricQuestion(metric=MetricName.TEXT_STATS, on="output"),),
    )
    with pytest.raises(ValidationError):
        _record(
            metric=MetricName.AST_STATS,
            on_key="nowhere",
            metrics_definition=elsewhere,
        )


def test_record_on_key_must_match_the_questions_on_key() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import (
        MetricName,
        MetricQuestion,
        MetricsDefinition,
    )

    definition = MetricsDefinition(
        definition_id="def",
        version="1",
        questions=(MetricQuestion(metric=MetricName.TEXT_STATS, on="output"),),
    )
    with pytest.raises(ValidationError):
        _record(
            metric=MetricName.TEXT_STATS,
            on_key="input",
            metrics_definition=definition,
        )


def test_record_settings_must_match_the_questions_settings() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import (
        MetricName,
        MetricQuestion,
        MetricsDefinition,
    )

    definition = MetricsDefinition(
        definition_id="def",
        version="1",
        questions=(
            MetricQuestion(
                metric=MetricName.CODE_LEAKAGE,
                on="input",
                settings={"task_names": ["declared"]},
            ),
        ),
    )
    with pytest.raises(ValidationError):
        _record(
            metric=MetricName.CODE_LEAKAGE,
            settings={"task_names": ["other"]},
            metrics_definition=definition,
        )


def test_record_matches_a_question_among_several() -> None:
    from dr_code.metrics import MetricName, MetricQuestion, MetricsDefinition

    definition = MetricsDefinition(
        definition_id="def",
        version="1",
        questions=(
            MetricQuestion(metric=MetricName.AST_STATS, on="output"),
            MetricQuestion(metric=MetricName.TEXT_STATS, on="input"),
        ),
    )
    record = _record(
        metric=MetricName.TEXT_STATS,
        on_key="input",
        metrics_definition=definition,
    )
    assert record.metric is MetricName.TEXT_STATS


def test_engine_produced_records_satisfy_the_identity_rule(text_trace) -> None:
    """Whatever the engine emits is loadable; the rule mirrors its derivation."""
    for record in _engine_records(text_trace):
        assert record == type(record).model_validate_json(
            record.model_dump_json()
        )


# ===========================================================================
# Serialization boundary: every status shape, non-trivial settings.
# ===========================================================================


def _engine_records(text_trace) -> tuple[object, ...]:
    from dr_code.metrics import MetricName, MetricQuestion, MetricsDefinition
    from dr_code.metrics.engine.engine import extract_metrics

    definition = MetricsDefinition(
        definition_id="round-trip",
        version="1",
        questions=(
            MetricQuestion(
                metric=MetricName.COMPRESSED_LENGTH,
                on="output",
                settings={"compression": {"method": "gzip", "level": 9}},
            ),
            MetricQuestion(
                metric=MetricName.CODE_LEAKAGE,
                on="output",
                settings={"task_names": ["add_one"]},
            ),
        ),
    )
    return extract_metrics(definition, text_trace("hello world"))


_ROUND_TRIP_SETTINGS = (
    pytest.param(
        "compressed_length",
        {"compression": {"method": "gzip", "level": 9}},
        "CompressedLengthSettings",
        id="compressed_length-gzip",
    ),
    pytest.param(
        "compressed_length",
        {"compression": {"method": "zstd", "level": 3}},
        "CompressedLengthSettings",
        id="compressed_length-zstd",
    ),
    pytest.param(
        "code_leakage",
        {"task_names": ["alpha", "beta"]},
        "CodeLeakageSettings",
        id="code_leakage-task-names",
    ),
)


def _shape_overrides(status: object) -> dict[str, object]:
    from dr_code.metrics import RecordStatus

    if status is RecordStatus.MEASURED:
        return {"values": {"count": 1}}
    if status is RecordStatus.NOT_APPLICABLE:
        return {
            "values": {},
            "absence_failed_step": "extract",
            "absence_cause": "no code extracted",
        }
    return {
        "values": {},
        "failure_type": "ValueError",
        "failure_message": "boom",
    }


@pytest.mark.parametrize(
    "metric_value,settings,settings_class", _ROUND_TRIP_SETTINGS
)
def test_record_round_trips_every_status_shape_with_registry_settings(
    metric_value: str, settings: dict[str, object], settings_class: str
) -> None:
    from dr_code.metrics import MetricName, MetricRecord, RecordStatus
    from dr_code.metrics.operators.base import OperatorSettings

    metric = MetricName(metric_value)
    for status in RecordStatus:
        record = _record(
            metric=metric,
            settings=settings,
            status=status,
            **_shape_overrides(status),
        )
        restored = MetricRecord.model_validate_json(record.model_dump_json())
        assert restored == record
        assert type(restored.settings).__name__ == settings_class
        assert type(restored.settings) is not OperatorSettings
        assert restored.settings == record.settings
