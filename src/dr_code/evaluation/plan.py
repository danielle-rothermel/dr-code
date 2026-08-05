"""The executable declaration of one evaluation.

An ``EvaluationPlan`` is a complete statement of what to evaluate (a task
set), how many times (a repeat plan), how to turn each response into
measurements (a procedure), and how to reduce those measurements to a single
number (an aggregation policy). It is a declaration, not a result: nothing
here records what happened.

A *procedure* nests resolved definitions rather than coordinates. That is
the one place in this package where registry-coupled validation is correct:
a procedure declares work that is about to run, so a definition it names had
better resolve now. Archived coordinates are the opposite case — they are
read long after the registry moved on — which is why the coordinates module
carries projections instead. Nothing in ``dr_code.evaluation`` performs a
registry lookup itself; the nested definitions do their own on validation.
"""

from __future__ import annotations

from enum import StrEnum, verify, UNIQUE
from typing import Self

from pydantic import model_validator

from dr_code.base import FrozenModel
from dr_code.evaluation.coordinates import RepeatPlan, TaskSet
from dr_code.metrics import MetricQuestionCoordinate, MetricsDefinition
from dr_code.preprocessing import PreprocessingDefinition


class EvaluationProcedure(FrozenModel):
    """How one response becomes measurements.

    Preprocessing turns a raw response into candidates; the metrics
    definition asks its questions of the resulting trace. Both are nested
    whole, so a procedure is executable on its own without a lookup.
    """

    preprocessing: PreprocessingDefinition
    metrics: MetricsDefinition
    __hash__ = None


@verify(UNIQUE)
class AggregationStatistic(StrEnum):
    """Every statistic an aggregation policy may compute.

    Never build a payload by iterating this enum: the members are a closed
    vocabulary, not an ordered list, and iteration order is not part of any
    persisted format. Reference members individually by name.
    """

    #: The arithmetic mean of the counted fact values.
    MEAN = "mean"
    #: The total of the counted fact values.
    SUM = "sum"
    #: How many slots contributed a counted value.
    COUNT = "count"
    #: The share of counted values that are truthy, on a zero-to-one scale.
    PROPORTION = "proportion"


@verify(UNIQUE)
class NotApplicablePolicy(StrEnum):
    """What a not-applicable slot contributes to the aggregate.

    A not-applicable record is a real answer — the question had no input —
    so how it counts is a judgment the policy must state rather than a
    default the aggregator picks. Never build a payload by iterating this
    enum.
    """

    #: The slot leaves the denominator entirely, as if never planned.
    EXCLUDE = "exclude"
    #: The slot counts in the denominator contributing a value of zero.
    ZERO = "zero"
    #: A not-applicable slot invalidates the aggregate.
    FAIL = "fail"


class AggregationPolicy(FrozenModel):
    """Which fact to reduce, by which statistic, counting which slots.

    The surface is exactly what reducing measurements to one number needs
    and nothing more:

    - ``question`` and ``fact`` name *which* number to read. A fact's full
      coordinate is its question coordinate plus its name, so the pair is
      the minimal complete address.
    - ``statistic`` names the reduction.
    - ``not_applicable`` and ``operator_failure`` state the denominator
      rules for the two non-measured record kinds. They are separate
      because they are genuinely different events — a question that had no
      input is not an operator that raised — and an evaluation may
      reasonably count one and refuse the other.

    Missing slots are deliberately not a knob. A slot the plan expected and
    that carries no record at all means the evaluation is incomplete, which
    is a fact about the run rather than a statistic to configure, so
    ``aggregate`` always reports it rather than letting a policy suppress
    it.
    """

    question: MetricQuestionCoordinate
    fact: str
    statistic: AggregationStatistic
    not_applicable: NotApplicablePolicy = NotApplicablePolicy.EXCLUDE
    operator_failure: NotApplicablePolicy = NotApplicablePolicy.FAIL

    @model_validator(mode="after")
    def validate_fact_name(self) -> Self:
        if not self.fact:
            raise ValueError("an aggregation policy must name a fact")
        return self


class EvaluationPlan(FrozenModel):
    """A complete, self-contained declaration of one evaluation.

    The task set and repeat plan must agree on size: the plan's
    ``task_count`` is the number of selected tasks, since the repeat plan's
    task-major slot layout is defined over exactly that selection.
    """

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
        """The aggregated question must be one the procedure asks.

        Scoped to internal consistency, exactly as record identity is: the
        check is against the plan's own nested metrics definition, never
        against a registry.
        """

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
