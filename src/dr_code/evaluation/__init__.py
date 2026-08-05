"""Evaluation plans, coordinates, and pure aggregation.

An evaluation declares what to measure and reduces the measurements to a
score. Three layers, kept separate on purpose:

``coordinates``
    Registry-free addresses — dataset, task set, repeat plan, sample,
    candidate — that persisted artifacts carry so they stay interpretable
    after the things they name have moved on.

``plan``
    The executable declaration: a procedure nesting resolved preprocessing
    and metrics definitions, an aggregation policy, and the plan tying them
    to a task set and repeat plan.

``aggregation`` and ``score``
    The pure reduction and the derived value it produces. ``aggregate``
    performs no I/O, consults no registry, and reads no clock.

This package is producer-blind and executor-blind. It never looks anything
up in a registry — the plan's nested definitions validate themselves — and
it knows nothing about any specific dataset or benchmark.
"""

from __future__ import annotations

from dr_code.evaluation.aggregation import (
    AggregationEmptyDenominator,
    AggregationInput,
    AggregationMissing,
    AggregationNonFinite,
    AggregationNotApplicable,
    AggregationOk,
    AggregationResult,
    AggregationSlot,
    AggregationStatus,
    FactCoordinate,
    aggregate,
)
from dr_code.evaluation.coordinates import (
    CandidateCoordinate,
    DatasetCoordinate,
    RepeatPlan,
    RepeatPlanCoordinate,
    SampleCoordinate,
    TaskSet,
    TaskSetCoordinate,
)
from dr_code.evaluation.plan import (
    AggregationPolicy,
    AggregationStatistic,
    EvaluationPlan,
    EvaluationProcedure,
    NotApplicablePolicy,
)
from dr_code.evaluation.score import EvaluationCoordinate, Score

__all__ = [
    "AggregationEmptyDenominator",
    "AggregationInput",
    "AggregationMissing",
    "AggregationNonFinite",
    "AggregationNotApplicable",
    "AggregationOk",
    "AggregationPolicy",
    "AggregationResult",
    "AggregationSlot",
    "AggregationStatistic",
    "AggregationStatus",
    "CandidateCoordinate",
    "DatasetCoordinate",
    "EvaluationCoordinate",
    "EvaluationPlan",
    "EvaluationProcedure",
    "FactCoordinate",
    "NotApplicablePolicy",
    "RepeatPlan",
    "RepeatPlanCoordinate",
    "SampleCoordinate",
    "Score",
    "TaskSet",
    "TaskSetCoordinate",
    "aggregate",
]
