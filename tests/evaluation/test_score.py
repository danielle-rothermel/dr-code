from __future__ import annotations

import pytest
from pydantic import ValidationError

from _builders import (
    question_coordinate,
    repeat_plan_coordinate,
    task_set_coordinate,
)
from dr_code.evaluation import (
    EvaluationCoordinate,
    FactCoordinate,
    Score,
)
from dr_code.metrics import MetricFactUnit


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


def fact_coordinate(**overrides: object) -> FactCoordinate:
    return FactCoordinate(
        **{
            "question": question_coordinate(),
            "fact": "char_count",
            **overrides,
        }
    )


def score(**overrides: object) -> Score:
    return Score(
        **{
            "name": "mean_char_count",
            "value": 12.5,
            "unit": MetricFactUnit.COUNT,
            "evaluation": evaluation_coordinate(),
            "sources": (fact_coordinate(),),
            **overrides,
        }
    )


def test_score_reuses_the_metric_fact_unit_vocabulary() -> None:
    assert Score.model_fields["unit"].annotation is MetricFactUnit


def test_score_rejects_the_text_unit() -> None:
    with pytest.raises(ValidationError, match="a score is a measurement"):
        score(unit=MetricFactUnit.TEXT)


def test_score_rejects_the_text_unit_by_its_wire_value() -> None:
    with pytest.raises(ValidationError, match="a score is a measurement"):
        score(unit="text")


@pytest.mark.parametrize(
    "unit",
    [unit for unit in MetricFactUnit if unit is not MetricFactUnit.TEXT],
)
def test_score_accepts_every_measurement_unit(unit: MetricFactUnit) -> None:
    assert score(unit=unit).unit is unit


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_score_rejects_a_non_finite_value(value: float) -> None:
    with pytest.raises(ValidationError, match="must be a finite value"):
        score(value=value)


def test_score_accepts_zero() -> None:
    assert score(value=0.0).value == 0.0


def test_score_requires_at_least_one_source_fact() -> None:
    with pytest.raises(ValidationError, match="must name the facts"):
        score(sources=())


def test_score_rejects_duplicated_source_facts() -> None:
    with pytest.raises(ValidationError, match="source facts must be unique"):
        score(sources=(fact_coordinate(), fact_coordinate()))


def test_score_records_distinct_facts_of_the_same_question() -> None:
    built = score(
        sources=(
            fact_coordinate(fact="char_count"),
            fact_coordinate(fact="word_count"),
        )
    )
    assert {source.fact for source in built.sources} == {
        "char_count",
        "word_count",
    }


def test_a_fact_coordinate_is_a_question_plus_a_name() -> None:
    assert set(FactCoordinate.model_fields) == {"question", "fact"}


def test_a_fact_coordinate_rejects_an_empty_fact_name() -> None:
    with pytest.raises(ValidationError, match="must name a fact"):
        fact_coordinate(fact="")


@pytest.mark.parametrize("name", ("x.unit", "char_count.unit", "a.b"))
def test_a_fact_coordinate_rejects_a_dotted_fact_name(name: str) -> None:
    with pytest.raises(ValidationError, match="must not contain"):
        fact_coordinate(fact=name)


def test_score_is_not_accepted_as_a_metric_fact() -> None:
    from dr_code.metrics import MetricFact

    with pytest.raises(ValidationError):
        MetricFact(
            name="mean_char_count",
            value=score(),
            unit=MetricFactUnit.COUNT,
        )


@pytest.mark.parametrize(
    "value",
    [evaluation_coordinate(), fact_coordinate(), score()],
)
def test_score_model_round_trips_through_json(value) -> None:
    assert type(value).model_validate_json(value.model_dump_json()) == value
