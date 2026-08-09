from __future__ import annotations

from enum import StrEnum, verify, UNIQUE
from typing import Self

from pydantic import model_validator

from dr_code.core.models import FrozenModel
from dr_code.evaluation.coordinates import RepeatPlan, TaskSet
from dr_code.metrics import MetricQuestionCoordinate, MetricsDefinition
from dr_code.preprocessing import PreprocessingDefinition


class EvaluationProcedure(FrozenModel):
    preprocessing: PreprocessingDefinition
    metrics: MetricsDefinition
    __hash__ = None


@verify(UNIQUE)
class AggregationStatistic(StrEnum):
    # Never build payloads by iterating this closed vocabulary.

    MEAN = "mean"
    SUM = "sum"
    COUNT = "count"
    PROPORTION = "proportion"


@verify(UNIQUE)
class NotApplicablePolicy(StrEnum):
    # Never build payloads by iterating this closed vocabulary.

    EXCLUDE = "exclude"
    ZERO = "zero"
    FAIL = "fail"


class AggregationPolicy(FrozenModel):
    question: MetricQuestionCoordinate
    value: str
    statistic: AggregationStatistic
    not_applicable: NotApplicablePolicy = NotApplicablePolicy.EXCLUDE
    operator_failure: NotApplicablePolicy = NotApplicablePolicy.FAIL

    @model_validator(mode="after")
    def validate_value_name(self) -> Self:
        if not self.value:
            raise ValueError("an aggregation policy must name a metric value")
        if "." in self.value:
            raise ValueError(
                f"metric value name {self.value!r} must not contain '.': "
                "no metric value can carry a dotted name"
            )
        return self


class EvaluationPlan(FrozenModel):
    plan_id: str
    version: str
    task_set: TaskSet
    repeat_plan: RepeatPlan
    procedure: EvaluationProcedure
    aggregation: AggregationPolicy
    __hash__ = None

    @model_validator(mode="after")
    def validate_plan_covers_the_selection(self) -> Self:
        selected = len(self.task_set.selected)
        if self.repeat_plan.task_count != selected:
            raise ValueError(
                "the repeat plan must cover exactly the selected tasks: "
                f"plan covers {self.repeat_plan.task_count}, task set "
                f"selects {selected}"
            )
        return self

    @model_validator(mode="after")
    def validate_aggregation_names_a_declared_question(self) -> Self:
        declared = tuple(
            MetricQuestionCoordinate.of(question)
            for question in self.procedure.metrics.questions
        )
        if self.aggregation.question not in declared:
            raise ValueError(
                f"aggregation names metric {self.aggregation.question.metric}"
                f" on {self.aggregation.question.on_key!r}, which the "
                "procedure's metrics definition does not declare"
            )
        return self


__all__ = [
    "AggregationPolicy",
    "AggregationStatistic",
    "EvaluationPlan",
    "EvaluationProcedure",
    "NotApplicablePolicy",
]
