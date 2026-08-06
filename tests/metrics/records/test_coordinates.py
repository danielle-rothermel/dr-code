from __future__ import annotations

import pytest

from ._builders import _identity, _question_coordinate


def test_identity_carries_the_question_and_both_coordinates() -> None:
    identity = _identity()
    assert identity.question.metric.value == "text_stats"
    assert identity.question.on_key == "input"
    assert identity.metric_version == "1"
    assert identity.producer.kind == "preprocessing"
    assert identity.producer.definition.definition_id == "pre"
    assert identity.producer.definition.version == "v1"
    assert identity.metrics_definition.definition_id == "def"
    assert identity.metrics_definition.version == "1"
    assert identity.metrics_definition.questions == (identity.question,)


def test_identity_must_name_a_question_its_definition_declares() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import MetricName, MetricsDefinitionCoordinate

    elsewhere = MetricsDefinitionCoordinate(
        definition_id="def",
        version="1",
        questions=(
            _question_coordinate(
                metric=MetricName.TEXT_STATS, on_key="output"
            ),
        ),
    )
    with pytest.raises(ValidationError):
        _identity(
            question=_question_coordinate(metric=MetricName.AST_STATS),
            metrics_definition=elsewhere,
        )


def test_identity_on_key_must_match_the_declared_question() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import MetricsDefinitionCoordinate

    definition = MetricsDefinitionCoordinate(
        definition_id="def",
        version="1",
        questions=(_question_coordinate(on_key="output"),),
    )
    with pytest.raises(ValidationError):
        _identity(
            question=_question_coordinate(on_key="input"),
            metrics_definition=definition,
        )


def test_identity_settings_must_match_the_declared_question() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import MetricName, MetricsDefinitionCoordinate
    from dr_code.trace import ComponentSetting

    declared = _question_coordinate(
        metric=MetricName.CODE_LEAKAGE,
        settings=(ComponentSetting(name="task_names", value=("declared",)),),
    )
    definition = MetricsDefinitionCoordinate(
        definition_id="def",
        version="1",
        questions=(declared,),
    )
    with pytest.raises(ValidationError):
        _identity(
            question=_question_coordinate(
                metric=MetricName.CODE_LEAKAGE,
                settings=(
                    ComponentSetting(name="task_names", value=("other",)),
                ),
            ),
            metrics_definition=definition,
        )


def test_identity_matches_a_question_among_several() -> None:
    from dr_code.metrics import MetricName, MetricsDefinitionCoordinate

    question = _question_coordinate(metric=MetricName.TEXT_STATS)
    definition = MetricsDefinitionCoordinate(
        definition_id="def",
        version="1",
        questions=(
            _question_coordinate(metric=MetricName.AST_STATS, on_key="output"),
            question,
        ),
    )
    identity = _identity(question=question, metrics_definition=definition)
    assert identity.question is question


def test_definition_coordinate_rejects_duplicate_questions() -> None:
    from pydantic import ValidationError

    from dr_code.metrics import MetricsDefinitionCoordinate

    question = _question_coordinate()
    with pytest.raises(ValidationError):
        MetricsDefinitionCoordinate(
            definition_id="def",
            version="1",
            questions=(question, question),
        )


def test_question_coordinate_projects_a_declared_questions_settings() -> None:
    from dr_code.metrics import (
        MetricName,
        MetricQuestion,
        MetricQuestionCoordinate,
    )
    from dr_code.trace import ComponentSetting

    coordinate = MetricQuestionCoordinate.of(
        MetricQuestion(
            metric=MetricName.CODE_LEAKAGE,
            on="output",
            settings={"task_names": ["x", "y"]},
        )
    )
    assert coordinate.metric is MetricName.CODE_LEAKAGE
    assert coordinate.on_key == "output"
    assert coordinate.settings == (
        ComponentSetting(name="task_names", value=("x", "y")),
    )
