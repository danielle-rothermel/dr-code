from __future__ import annotations

import pytest

from dr_code.evaluation import (
    AggregationPolicy,
    AggregationSlot,
    AggregationStatistic,
    EvaluationCandidateIdentity,
    EvaluationSampleIdentity,
    EvaluationSlotIdentity,
    DatasetCoordinate,
    EvaluationProcedure,
    SamplingPlan,
    SamplingPlanCoordinate,
    TaskSet,
    TaskSetCoordinate,
)
from dr_code.metrics import (
    MeasuredRecord,
    MetricValue,
    MetricValueUnit,
    MetricName,
    MetricQuestion,
    MetricQuestionCoordinate,
    MetricRecordIdentity,
    MetricsDefinition,
    MetricsDefinitionCoordinate,
    NotApplicableRecord,
    OperatorFailure,
    OperatorFailureRecord,
)
from dr_code.preprocessing import PreprocessingDefinition, StepName, StepSpec
from dr_code.trace import (
    Absent,
    ComponentCoordinate,
    PreprocessingDefinitionCoordinate,
    PreprocessingTraceProducer,
    StepCoordinate,
)

TASK_SET_ID = "task-set"
SAMPLING_PLAN_ID = "sampling-plan"
PLAN_ID = "plan"
VALUE_NAME = "char_count"


def dataset(**overrides: object) -> DatasetCoordinate:
    return DatasetCoordinate(
        **{"dataset_id": "dataset", "version": "1", **overrides}
    )


def task_set_coordinate(**overrides: object) -> TaskSetCoordinate:
    return TaskSetCoordinate(
        **{
            "task_set_id": TASK_SET_ID,
            "version": "1",
            "dataset": dataset(),
            **overrides,
        }
    )


def task_set(**overrides: object) -> TaskSet:
    return TaskSet(
        **{
            "coordinate": task_set_coordinate(),
            "population": ("t0", "t1", "t2"),
            "selected": ("t0", "t2"),
            **overrides,
        }
    )


def sampling_plan_coordinate(**overrides: object) -> SamplingPlanCoordinate:
    return SamplingPlanCoordinate(
        **{"sampling_plan_id": SAMPLING_PLAN_ID, "version": "1", **overrides}
    )


def sampling_plan(**overrides: object) -> SamplingPlan:
    return SamplingPlan(
        **{
            "coordinate": sampling_plan_coordinate(),
            "task_count": 2,
            "task_num_samples": (2, 2),
            **overrides,
        }
    )


def evaluation_slot(**overrides: object) -> EvaluationSlotIdentity:
    return EvaluationSlotIdentity(
        **{
            "task_set": task_set_coordinate(),
            "sampling_plan": sampling_plan_coordinate(),
            "task_id": "t0",
            "sample_index": 0,
            **overrides,
        }
    )


def preprocessing_coordinate(
    **overrides: object,
) -> PreprocessingDefinitionCoordinate:
    return PreprocessingDefinitionCoordinate(
        **{
            "definition_id": "pre",
            "version": "0",
            "steps": (
                StepCoordinate(
                    instance_name="normalize",
                    component=ComponentCoordinate(
                        registered_name=StepName.NORMALIZE_UNICODE.value,
                        version="0",
                    ),
                ),
            ),
            **overrides,
        }
    )


def sample_identity(**overrides: object) -> EvaluationSampleIdentity:
    return EvaluationSampleIdentity(**{"sample_id": "sample-0", **overrides})


def candidate(**overrides: object) -> EvaluationCandidateIdentity:
    return EvaluationCandidateIdentity(
        **{
            "sample": sample_identity(),
            "preprocessing": preprocessing_coordinate(),
            "candidate_ordinal": 0,
            **overrides,
        }
    )


def preprocessing_definition(**overrides: object) -> PreprocessingDefinition:
    return PreprocessingDefinition(
        **{
            "definition_id": "pre",
            "version": "0",
            "steps": (
                StepSpec(
                    instance_name="normalize",
                    step=StepName.NORMALIZE_UNICODE,
                ),
            ),
            **overrides,
        }
    )


def metrics_definition(**overrides: object) -> MetricsDefinition:
    return MetricsDefinition(
        **{
            "definition_id": "met",
            "version": "0",
            "questions": (
                MetricQuestion(metric=MetricName.TEXT_STATS, on="output"),
            ),
            **overrides,
        }
    )


def question_coordinate(**overrides: object) -> MetricQuestionCoordinate:
    return MetricQuestionCoordinate(
        **{
            "metric": MetricName.TEXT_STATS,
            "on_key": "output",
            "settings": (),
            **overrides,
        }
    )


def procedure(**overrides: object) -> EvaluationProcedure:
    return EvaluationProcedure(
        **{
            "preprocessing": preprocessing_definition(),
            "metrics": metrics_definition(),
            **overrides,
        }
    )


def policy(**overrides: object) -> AggregationPolicy:
    return AggregationPolicy(
        **{
            "question": question_coordinate(),
            "value": VALUE_NAME,
            "statistic": AggregationStatistic.MEAN,
            **overrides,
        }
    )


def record_identity(**overrides: object) -> MetricRecordIdentity:
    question = overrides.pop("question", question_coordinate())
    return MetricRecordIdentity(
        **{
            "question": question,
            "metric_version": "0",
            "producer": PreprocessingTraceProducer(
                definition=preprocessing_coordinate()
            ),
            "metrics_definition": MetricsDefinitionCoordinate(
                definition_id="met",
                version="0",
                questions=(question,),
            ),
            **overrides,
        }
    )


def measured(
    value: float | int | bool = 1,
    *,
    name: str = VALUE_NAME,
    unit: MetricValueUnit = MetricValueUnit.COUNT,
    **overrides: object,
) -> MeasuredRecord:
    return MeasuredRecord(
        **{
            "identity": record_identity(),
            "values": (MetricValue(name=name, value=value, unit=unit),),
            **overrides,
        }
    )


def not_applicable(**overrides: object) -> NotApplicableRecord:
    return NotApplicableRecord(
        **{
            "identity": record_identity(),
            "absence": Absent(
                failed_step="normalize",
                failure_code="blank_input",
                cause="nothing to measure",
            ),
            **overrides,
        }
    )


def operator_failure(**overrides: object) -> OperatorFailureRecord:
    return OperatorFailureRecord(
        **{
            "identity": record_identity(),
            "failure": OperatorFailure(
                failure_type="ValueError", failure_message="boom"
            ),
            **overrides,
        }
    )


def slot(record: object = None, *, ordinal: int = 0) -> AggregationSlot:
    return AggregationSlot(
        candidate=candidate(candidate_ordinal=ordinal), record=record
    )


@pytest.fixture
def mean_policy() -> AggregationPolicy:
    return policy()
