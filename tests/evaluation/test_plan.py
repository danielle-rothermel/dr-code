from __future__ import annotations

import pytest
from pydantic import ValidationError

from _builders import (
    metrics_definition,
    policy,
    preprocessing_definition,
    procedure,
    question_coordinate,
    repeat_plan,
    task_set,
)
from dr_code.evaluation import (
    AggregationPolicy,
    AggregationStatistic,
    EvaluationPlan,
    EvaluationProcedure,
    NotApplicablePolicy,
)
from dr_code.metrics import (
    MetricName,
    MetricQuestion,
    MetricsDefinition,
)

EXPECTED_STATISTICS = {"mean", "sum", "count", "proportion"}
EXPECTED_NOT_APPLICABLE_POLICIES = {"exclude", "zero", "fail"}


def evaluation_plan(**overrides: object) -> EvaluationPlan:
    return EvaluationPlan(
        **{
            "plan_id": "plan",
            "version": "1",
            "task_set": task_set(),
            "repeat_plan": repeat_plan(),
            "procedure": procedure(),
            "aggregation": policy(),
            **overrides,
        }
    )


def test_statistic_vocabulary_is_closed() -> None:
    assert {
        member.value for member in AggregationStatistic
    } == EXPECTED_STATISTICS


def test_not_applicable_policy_vocabulary_is_closed() -> None:
    assert {
        member.value for member in NotApplicablePolicy
    } == EXPECTED_NOT_APPLICABLE_POLICIES


def test_procedure_nests_whole_resolved_definitions() -> None:
    built = procedure()
    assert built.preprocessing == preprocessing_definition()
    assert built.metrics == metrics_definition()
    assert built.preprocessing.steps[0].instance_name == "normalize"


def test_procedure_rejects_a_definition_coordinate_in_place_of_a_definition() -> (
    None
):
    with pytest.raises(ValidationError):
        EvaluationProcedure(
            preprocessing={
                "definition_id": "pre",
                "version": "0",
                "steps": ({"instance_name": "x", "component": {}},),
            },
            metrics=metrics_definition(),
        )


def test_policy_defaults_exclude_not_applicable_and_fail_on_error() -> None:
    built = policy()
    assert built.not_applicable is NotApplicablePolicy.EXCLUDE
    assert built.operator_failure is NotApplicablePolicy.FAIL


def test_policy_rejects_an_empty_value_name() -> None:
    with pytest.raises(ValidationError, match="must name a metric value"):
        policy(value="")


@pytest.mark.parametrize("name", ("x.unit", "char_count.unit"))
def test_policy_rejects_a_dotted_value_name(name: str) -> None:
    with pytest.raises(ValidationError, match="must not contain"):
        policy(value=name)


def test_policy_fields_are_exactly_the_minimal_surface() -> None:
    assert set(AggregationPolicy.model_fields) == {
        "question",
        "value",
        "statistic",
        "not_applicable",
        "operator_failure",
    }


def test_policy_rejects_an_unknown_knob() -> None:
    with pytest.raises(ValidationError):
        policy(threshold=0.5)


def test_plan_accepts_a_repeat_plan_covering_the_selection() -> None:
    built = evaluation_plan()
    assert built.repeat_plan.task_count == len(built.task_set.selected)


def test_plan_rejects_a_repeat_plan_covering_too_few_tasks() -> None:
    with pytest.raises(ValidationError, match="exactly the selected tasks"):
        evaluation_plan(repeat_plan=repeat_plan(task_count=1))


def test_plan_rejects_a_repeat_plan_covering_too_many_tasks() -> None:
    with pytest.raises(ValidationError, match="exactly the selected tasks"):
        evaluation_plan(repeat_plan=repeat_plan(task_count=3))


def test_plan_rejects_aggregating_an_undeclared_question() -> None:
    undeclared = question_coordinate(on_key="never_declared")
    with pytest.raises(ValidationError, match="does not declare"):
        evaluation_plan(aggregation=policy(question=undeclared))


def test_plan_accepts_aggregating_any_declared_question() -> None:
    definition = MetricsDefinition(
        definition_id="met",
        version="0",
        questions=(
            MetricQuestion(metric=MetricName.TEXT_STATS, on="output"),
            MetricQuestion(metric=MetricName.AST_STATS, on="code"),
        ),
    )
    built = evaluation_plan(
        procedure=procedure(metrics=definition),
        aggregation=policy(
            question=question_coordinate(
                metric=MetricName.AST_STATS, on_key="code"
            )
        ),
    )
    assert built.aggregation.question.on_key == "code"


def test_plan_consistency_check_compares_against_its_own_definition() -> None:
    declared = policy(question=question_coordinate(on_key="output"))
    assert evaluation_plan(aggregation=declared).aggregation == declared

    elsewhere = MetricsDefinition(
        definition_id="met",
        version="0",
        questions=(
            MetricQuestion(metric=MetricName.TEXT_STATS, on="other_key"),
        ),
    )
    with pytest.raises(ValidationError, match="does not declare"):
        evaluation_plan(
            procedure=procedure(metrics=elsewhere), aggregation=declared
        )


def test_plan_load_resolves_definitions_through_the_registry() -> None:
    from types import MappingProxyType

    import dr_code.metrics.registry as registry_module

    payload = evaluation_plan().model_dump_json()
    original = registry_module.REGISTRY
    try:
        registry_module.REGISTRY = MappingProxyType({})
        with pytest.raises(KeyError):
            EvaluationPlan.model_validate_json(payload)
    finally:
        registry_module.REGISTRY = original


@pytest.mark.parametrize(
    "value",
    [
        procedure(),
        policy(),
        policy(
            statistic=AggregationStatistic.PROPORTION,
            not_applicable=NotApplicablePolicy.ZERO,
            operator_failure=NotApplicablePolicy.EXCLUDE,
        ),
        evaluation_plan(),
    ],
)
def test_plan_model_round_trips_through_json(value) -> None:
    assert type(value).model_validate_json(value.model_dump_json()) == value
