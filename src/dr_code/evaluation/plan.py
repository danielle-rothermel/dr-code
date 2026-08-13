from __future__ import annotations

from enum import StrEnum, verify, UNIQUE
from typing import Self

from pydantic import model_validator

from dr_code.core.models import FrozenModel
from dr_code.evaluation.coordinates import SamplingPlan, TaskSet
from dr_code.evaluation.identity import EvalSlotIdentity
from dr_code.metrics import MetricQuestionCoordinate, MetricsDefinition
from dr_code.preprocessing import PreprocessingDefinition


class EvalProcedure(FrozenModel):
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


class EvalPlan(FrozenModel):
    plan_id: str
    version: str
    task_set: TaskSet
    sampling_plan: SamplingPlan
    procedure: EvalProcedure
    aggregation: AggregationPolicy
    __hash__ = None

    def ordered_slots(self) -> tuple[EvalSlotIdentity, ...]:
        """Return every slot the plan declares, in plan order.

        Each selected task contributes exactly the samples the plan declares
        for it, so the slot sequence is the plan's exact expected membership.
        """

        return tuple(
            EvalSlotIdentity(
                task_set=self.task_set.coordinate,
                sampling_plan=self.sampling_plan.coordinate,
                task_id=task_id,
                sample_index=sample_index,
            )
            for task_index, task_id in enumerate(self.task_set.selected)
            for sample_index in range(
                self.sampling_plan.num_samples_for(task_index)
            )
        )

    def declares_slot(self, slot: EvalSlotIdentity) -> bool:
        """Report whether this plan declares the given slot position."""

        if (
            slot.task_set != self.task_set.coordinate
            or slot.sampling_plan != self.sampling_plan.coordinate
        ):
            return False
        try:
            task_index = self.task_set.selected.index(slot.task_id)
        except ValueError:
            return False
        num_samples = self.sampling_plan.num_samples_for(task_index)
        return 0 <= slot.sample_index < num_samples

    @model_validator(mode="after")
    def validate_plan_covers_the_selection(self) -> Self:
        selected = len(self.task_set.selected)
        if self.sampling_plan.task_count != selected:
            raise ValueError(
                "the sampling plan must cover exactly the selected tasks: "
                f"plan covers {self.sampling_plan.task_count}, task set "
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
    "EvalPlan",
    "EvalProcedure",
    "NotApplicablePolicy",
]
