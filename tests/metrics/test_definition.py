"""Metric-definition contracts.

Covers ``MetricQuestion`` / ``MetricsDefinition`` — frozen, equality-based
comparability, the unique ``(metric, on, settings)`` validator, and settings
as part of the explicit declaration.

"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_code.metrics.operators.code_leakage import CodeLeakageSettings
from dr_code.metrics.settings import OperatorSettings


def _question(**overrides: object):
    from dr_code.metrics import MetricName, MetricQuestion

    base: dict[str, object] = {
        "metric": MetricName.TEXT_STATS,
        "on": "input",
        "settings": {},
    }
    base.update(overrides)
    return MetricQuestion(**base)


def _definition(
    questions=None,
    **overrides: object,
):
    from dr_code.metrics import MetricsDefinition

    base: dict[str, object] = {
        "definition_id": "def",
        "version": "1",
        "questions": questions or (_question(),),
    }
    base.update(overrides)
    return MetricsDefinition(**base)


# ===========================================================================
# MetricQuestion.
# ===========================================================================


def test_metric_question_carries_metric_on_settings() -> None:
    question = _question()
    assert question.settings == OperatorSettings()
    assert question.on == "input"


def test_metric_question_defaults_empty_settings() -> None:
    from dr_code.metrics import MetricName, MetricQuestion

    question = MetricQuestion(metric=MetricName.TEXT_STATS, on="input")
    assert question.settings == OperatorSettings()


def test_metric_question_carries_a_settings_dict() -> None:
    from dr_code.metrics import MetricName, MetricQuestion

    question = MetricQuestion(
        metric=MetricName.CODE_LEAKAGE,
        on="selected",
        settings={"task_names": ["add_one", "HumanEval/0"]},
    )
    assert question.settings == CodeLeakageSettings(
        task_names=("add_one", "HumanEval/0")
    )


def test_metric_question_field_set_is_exactly_metric_on_settings() -> None:
    """Precise schema: questions carry only the three identity fields."""
    from dr_code.metrics.definition import MetricQuestion

    assert set(MetricQuestion.model_fields) == {"metric", "on", "settings"}


def test_metric_question_is_frozen() -> None:
    question = _question()
    with pytest.raises(ValidationError) as exc_info:
        question.on = "output"  # type: ignore[misc]
    error = exc_info.value.errors()[0]
    assert (error["type"], error["loc"]) == ("frozen_instance", ("on",))


def test_metric_questions_compare_equal_by_value() -> None:
    """Structured equality is the comparability contract."""
    assert _question() == _question()


def test_metric_questions_differ_on_settings() -> None:
    from dr_code.metrics import MetricName

    a = _question(
        metric=MetricName.COMPRESSED_LENGTH,
        settings={"compression": {"method": "gzip", "level": 9}},
    )
    b = _question(
        metric=MetricName.COMPRESSED_LENGTH,
        settings={"compression": {"method": "zstd", "level": 3}},
    )
    assert a != b


def test_metric_question_settings_are_order_independent() -> None:
    """Dict key ordering does not affect equality (settings are identity)."""
    from dr_code.metrics import MetricName, MetricQuestion

    a = MetricQuestion(
        metric=MetricName.COMPRESSED_LENGTH,
        on="input",
        settings={"compression": {"method": "gzip", "level": 9}},
    )
    b = MetricQuestion(
        metric=MetricName.COMPRESSED_LENGTH,
        on="input",
        settings={"compression": {"level": 9, "method": "gzip"}},
    )
    assert a == b


# ===========================================================================
# MetricsDefinition.
# ===========================================================================


def test_metrics_definition_carries_id_version_questions() -> None:
    from dr_code.metrics import MetricName

    definition = _definition(
        definition_id="humaneval-metrics",
        version="v1",
        questions=(_question(metric=MetricName.TEXT_STATS, on="input"),),
    )
    assert definition.definition_id == "humaneval-metrics"
    assert definition.version == "v1"
    assert len(definition.questions) == 1


def test_metrics_definition_field_set_is_exactly_id_version_questions() -> (
    None
):
    from dr_code.metrics.definition import MetricsDefinition

    assert set(MetricsDefinition.model_fields) == {
        "definition_id",
        "version",
        "questions",
    }


def test_metrics_definition_questions_is_a_tuple() -> None:
    definition = _definition(
        questions=(_question(on="input"), _question(on="output")),
    )
    assert isinstance(definition.questions, tuple)


def test_metrics_definition_is_frozen() -> None:
    definition = _definition()
    with pytest.raises(ValidationError) as exc_info:
        definition.version = "2"  # type: ignore[misc]
    error = exc_info.value.errors()[0]
    assert (error["type"], error["loc"]) == (
        "frozen_instance",
        ("version",),
    )


def test_metrics_definition_questions_are_required() -> None:
    """Definitions require an explicit question sequence."""
    from dr_code.metrics import MetricsDefinition

    with pytest.raises(ValidationError) as exc_info:
        MetricsDefinition(definition_id="def", version="1")  # type: ignore[call-arg]
    error = exc_info.value.errors()[0]
    assert (error["type"], error["loc"]) == ("missing", ("questions",))


def test_metrics_definitions_compare_equal_by_value() -> None:
    assert _definition() == _definition()


def test_metrics_definition_json_round_trip_is_lossless() -> None:
    from dr_code.metrics import MetricName
    from dr_code.metrics.definition import MetricsDefinition

    definition = _definition(
        questions=(
            _question(
                metric=MetricName.CODE_LEAKAGE,
                on="selected",
                settings={"task_names": ["add_one"]},
            ),
            _question(on="input"),
        ),
    )
    restored = MetricsDefinition.model_validate_json(
        definition.model_dump_json()
    )
    assert restored == definition
    assert restored.questions[0].settings == CodeLeakageSettings(
        task_names=("add_one",)
    )


# ---------------------------------------------------------------------------
# Uniqueness of (metric, on, settings) triples.
# ---------------------------------------------------------------------------


def test_duplicate_metric_on_settings_triple_is_rejected() -> None:
    """Distinct questions need a distinct triple; a duplicate is a wiring bug.

    Resolve the import before ``pytest.raises`` so an import failure cannot be
    mistaken for the expected validation failure.
    """
    from dr_code.metrics import MetricName  # noqa: F401 — resolve before assert

    assert _question().metric == MetricName.TEXT_STATS
    with pytest.raises(ValidationError) as exc_info:
        _definition(
            questions=(
                _question(),
                _question(),
            ),
        )
    error = exc_info.value.errors()[0]
    assert (error["type"], error["loc"]) == ("value_error", ())


def test_same_metric_different_on_key_is_allowed() -> None:
    definition = _definition(
        questions=(_question(on="input"), _question(on="output")),
    )
    assert len(definition.questions) == 2


def test_same_metric_on_key_different_settings_is_allowed() -> None:
    """Settings participate in identity: two codec levels are two questions."""
    from dr_code.metrics import MetricName

    definition = _definition(
        questions=(
            _question(
                metric=MetricName.COMPRESSED_LENGTH,
                settings={"compression": {"method": "gzip", "level": 6}},
            ),
            _question(
                metric=MetricName.COMPRESSED_LENGTH,
                settings={"compression": {"method": "gzip", "level": 9}},
            ),
        ),
    )
    assert len(definition.questions) == 2


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
