from __future__ import annotations

import pytest
from pydantic import ValidationError

from _builders import (
    question_coordinate,
    repeat_plan_coordinate,
    task_set_coordinate,
)
from dr_code.evaluation import EvaluationCoordinate, Score
from dr_code.metrics import MetricValue, MetricValueCoordinate, MetricValueUnit


def evaluation_coordinate(**overrides: object) -> EvaluationCoordinate:
    return EvaluationCoordinate(
        **{
            "plan_id": "plan",
            "version": "1",
            "task_set": task_set_coordinate(),
            "repeat_plan": repeat_plan_coordinate(),
            **overrides,
        }
    )


def value_coordinate(**overrides: object) -> MetricValueCoordinate:
    return MetricValueCoordinate(
        **{
            "question": question_coordinate(),
            "value": "char_count",
            **overrides,
        }
    )


def score(**overrides: object) -> Score:
    return Score(
        **{
            "name": "mean_char_count",
            "value": 12.5,
            "unit": MetricValueUnit.COUNT,
            "evaluation": evaluation_coordinate(),
            "sources": (value_coordinate(),),
            **overrides,
        }
    )


def test_score_reuses_the_metric_value_unit_vocabulary() -> None:
    assert Score.model_fields["unit"].annotation is MetricValueUnit


def test_score_rejects_the_text_unit() -> None:
    with pytest.raises(ValidationError, match="a score is a measurement"):
        score(unit=MetricValueUnit.TEXT)


def test_score_rejects_the_text_unit_by_its_wire_value() -> None:
    with pytest.raises(ValidationError, match="a score is a measurement"):
        score(unit="text")


@pytest.mark.parametrize(
    "unit",
    [unit for unit in MetricValueUnit if unit is not MetricValueUnit.TEXT],
)
def test_score_accepts_every_measurement_unit(unit: MetricValueUnit) -> None:
    assert score(unit=unit).unit is unit


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_score_rejects_a_non_finite_value(value: float) -> None:
    with pytest.raises(ValidationError, match="must be a finite value"):
        score(value=value)


def test_score_accepts_zero() -> None:
    assert score(value=0.0).value == 0.0


def test_score_requires_at_least_one_source_value() -> None:
    with pytest.raises(ValidationError, match="must name the metric values"):
        score(sources=())


def test_score_rejects_duplicated_source_values() -> None:
    with pytest.raises(
        ValidationError, match="source metric values must be unique"
    ):
        score(sources=(value_coordinate(), value_coordinate()))


def test_score_records_distinct_values_of_the_same_question() -> None:
    built = score(
        sources=(
            value_coordinate(value="char_count"),
            value_coordinate(value="word_count"),
        )
    )
    assert {source.value for source in built.sources} == {
        "char_count",
        "word_count",
    }


def test_a_value_coordinate_is_a_question_plus_a_name() -> None:
    assert set(MetricValueCoordinate.model_fields) == {"question", "value"}


def test_a_value_coordinate_rejects_an_empty_value_name() -> None:
    with pytest.raises(ValidationError, match="must name a value"):
        value_coordinate(value="")


@pytest.mark.parametrize("name", ("x.unit", "char_count.unit", "a.b"))
def test_a_value_coordinate_rejects_a_dotted_value_name(name: str) -> None:
    with pytest.raises(ValidationError, match="must not contain"):
        value_coordinate(value=name)


def test_score_is_not_accepted_as_a_metric_value() -> None:
    with pytest.raises(ValidationError):
        MetricValue(
            name="mean_char_count",
            value=score(),
            unit=MetricValueUnit.COUNT,
        )


@pytest.mark.parametrize(
    "value",
    [evaluation_coordinate(), value_coordinate(), score()],
)
def test_score_model_round_trips_through_json(value) -> None:
    assert type(value).model_validate_json(value.model_dump_json()) == value
