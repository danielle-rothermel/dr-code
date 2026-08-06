from __future__ import annotations

from ._builders import (
    _fact,
    _identity,
    _measured,
    _not_applicable,
    _operator_failure,
    _question_coordinate,
)


def test_record_rows_returns_one_row_per_record() -> None:
    from dr_code.metrics import MetricName, record_rows

    other = _question_coordinate(metric=MetricName.AST_STATS)
    rows = record_rows(
        [_measured(), _measured(identity=_identity(question=other))]
    )
    assert len(rows) == 2
    assert all(isinstance(row, dict) for row in rows)


def test_record_rows_empty_input_returns_empty_list() -> None:
    from dr_code.metrics import record_rows

    assert record_rows([]) == []


def test_record_rows_prefix_fact_columns_with_metric_and_name() -> None:
    from dr_code.metrics import record_rows

    row = record_rows(
        [
            _measured(
                facts=(
                    _fact(name="character_count", value=4),
                    _fact(name="word_count", value=1),
                )
            )
        ]
    )[0]
    assert row["text_stats.character_count"] == 4
    assert row["text_stats.word_count"] == 1

    assert "character_count" not in row
    assert "word_count" not in row


def test_record_rows_carry_each_facts_unit_in_a_sibling_column() -> None:
    from dr_code.metrics import MetricFactUnit, record_rows

    row = record_rows(
        [
            _measured(
                facts=(_fact(name="byte_count", unit=MetricFactUnit.BYTES),)
            )
        ]
    )[0]
    assert row["text_stats.byte_count.unit"] == MetricFactUnit.BYTES

    assert "text_stats.byte_count.bytes" not in row


def test_record_rows_include_identity_and_lineage_columns() -> None:
    from dr_code.metrics import MetricName, RecordStatus, record_rows

    row = record_rows([_measured()])[0]
    assert row["schema_version"] == 1
    assert row["metric"] == MetricName.TEXT_STATS
    assert row["metric_version"] == "1"
    assert row["on_key"] == "input"
    assert row["question_settings"] == ()
    assert row["producer"]["definition"]["definition_id"] == "pre"
    assert row["metrics_definition"]["definition_id"] == "def"
    assert row["metrics_definition"]["version"] == "1"
    assert row["status"] == RecordStatus.MEASURED


def test_record_rows_fact_columns_are_collision_free_across_metrics() -> None:
    from dr_code.metrics import MetricName, record_rows

    ast = _question_coordinate(metric=MetricName.AST_STATS)
    rows = record_rows(
        [
            _measured(facts=(_fact(name="count", value=1),)),
            _measured(
                identity=_identity(question=ast),
                facts=(_fact(name="count", value=2),),
            ),
        ]
    )
    assert rows[0]["text_stats.count"] == 1
    assert rows[1]["ast_stats.count"] == 2


def test_record_rows_status_column_distinguishes_absence_from_zero() -> None:
    from dr_code.metrics import RecordStatus, record_rows

    rows = record_rows(
        [
            _measured(facts=(_fact(name="count", value=0),)),
            _not_applicable(),
        ]
    )
    assert rows[0]["status"] == RecordStatus.MEASURED
    assert rows[0]["text_stats.count"] == 0
    assert rows[1]["status"] == RecordStatus.NOT_APPLICABLE
    assert "text_stats.count" not in rows[1]
    assert rows[1]["absence"]["failed_step"] == "extract"
    assert rows[1]["absence"]["failure_code"] == "no_candidates_extracted"
    assert rows[1]["absence"]["cause"] == "no code extracted"


def test_record_rows_carry_the_operator_failure_payload() -> None:
    from dr_code.metrics import record_rows

    row = record_rows([_operator_failure()])[0]
    assert row["failure"]["failure_type"] == "ValueError"
    assert row["failure"]["failure_message"] == "boom"


def test_record_rows_preserve_declaration_order() -> None:
    from dr_code.metrics import MetricName, record_rows

    rows = record_rows(
        [
            _measured(),
            _measured(
                identity=_identity(
                    question=_question_coordinate(
                        metric=MetricName.CODE_LEAKAGE
                    )
                )
            ),
            _measured(
                identity=_identity(
                    question=_question_coordinate(metric=MetricName.AST_STATS)
                )
            ),
        ]
    )
    assert [row["metric"] for row in rows] == [
        MetricName.TEXT_STATS,
        MetricName.CODE_LEAKAGE,
        MetricName.AST_STATS,
    ]
