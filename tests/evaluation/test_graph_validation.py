from __future__ import annotations

import pytest
from dr_exec import ExecutionPoolConfig
from dr_serialize import IdentityDocument
from pydantic import ValidationError

from _executor_stubs import importable_json_executor
from _builders import (
    candidate,
    evaluation_slot,
    measured,
    preprocessing_coordinate,
    record_identity,
    sample_identity,
)
from dr_code.evaluation import (
    CandidateJobCompleted,
    EvaluationMemberRecord,
    EvaluationRuntimeIdentity,
    EvaluatedSampleRecord,
    SampleEvaluationRecord,
)
from dr_code.evaluation._batch import _evaluate_batch_assembly
from dr_code.evaluation.validation import validate_sample_record_graph
from dr_code.metrics import MetricsDefinitionCoordinate
from dr_code.trace import PreprocessingTraceProducer

from .test_record_models import (
    attempt,
    evaluated,
    evaluation_plan,
    execution,
    materialized,
    reference,
    runtime,
)
from ._batch_builders import BatchStore, MemoryPlacement, cache, request


def _validate(record: SampleEvaluationRecord) -> None:
    validate_sample_record_graph(
        record,
        slot=record.slot,
        sample=record.sample.identity,
        plan=evaluation_plan(),
        runtime=runtime(),
        cache_namespace="evaluation-v1",
    )


def test_attempt_rejects_member_slots_outside_plan() -> None:
    with pytest.raises(ValidationError, match="belong to the evaluation plan"):
        attempt(
            members=(
                EvaluationMemberRecord(
                    slot=evaluation_slot(task_id="t1"),
                    sample=sample_identity(),
                    record=reference(),
                ),
            )
        )


def test_attempt_rejects_members_out_of_plan_order() -> None:
    with pytest.raises(ValidationError, match="preserve plan slot order"):
        attempt(
            members=(
                EvaluationMemberRecord(
                    slot=evaluation_slot(repeat_index=1),
                    sample=sample_identity(sample_id="sample-1"),
                    record=reference(1),
                ),
                EvaluationMemberRecord(
                    slot=evaluation_slot(repeat_index=0),
                    sample=sample_identity(sample_id="sample-0"),
                    record=reference(0),
                ),
            )
        )


def test_graph_rejects_missing_and_wrong_metric_evidence() -> None:
    with pytest.raises(ValueError, match="exactly cover"):
        _validate(evaluated(metrics=()))

    wrong_definition = MetricsDefinitionCoordinate(
        definition_id="other-metrics",
        version="0",
        questions=(record_identity().question,),
    )
    wrong_metric = measured(
        identity=record_identity(metrics_definition=wrong_definition)
    )
    with pytest.raises(ValueError, match="definition does not match"):
        _validate(evaluated(metrics=(wrong_metric,)))


def test_graph_rejects_candidate_trace_and_execution_context_mismatches() -> (
    None
):
    ordinal_one = candidate(candidate_ordinal=1)
    with pytest.raises(ValueError, match="contiguous ordinal"):
        _validate(
            evaluated(
                candidates=(materialized(identity=ordinal_one),),
                executions=(execution(candidate=ordinal_one),),
            )
        )

    wrong_preprocessing = preprocessing_coordinate(
        definition_id="other-preprocessing"
    )
    record = evaluated()
    wrong_trace = record.trace.model_copy(
        update={
            "producer": PreprocessingTraceProducer(
                definition=wrong_preprocessing
            )
        }
    )
    with pytest.raises(ValueError, match="preprocessing does not match"):
        _validate(record.model_copy(update={"trace": wrong_trace}))

    wrong_runtime = EvaluationRuntimeIdentity(
        document=IdentityDocument(
            schema="tests/other-runtime",
            schema_version=1,
            payload={},
        )
    )
    with pytest.raises(ValueError, match="runtime does not match"):
        _validate(evaluated(executions=(execution(runtime=wrong_runtime),)))

    with pytest.raises(ValueError, match="cache namespace does not match"):
        _validate(evaluated(executions=(execution(cache_namespace="other"),)))

    wrong_candidate = candidate(preprocessing=wrong_preprocessing)
    with pytest.raises(ValueError, match="candidate identity does not match"):
        _validate(
            evaluated(
                candidates=(materialized(identity=wrong_candidate),),
                executions=(execution(candidate=wrong_candidate),),
            )
        )


@pytest.mark.asyncio
async def test_graph_rejects_completed_result_candidate_mismatch() -> None:
    batch_request = request(projections=())
    execution_cache = cache(BatchStore())
    placement = MemoryPlacement()
    await _evaluate_batch_assembly(
        batch_request,
        executor=importable_json_executor(),
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(),
        placement_sink=placement,
    )
    record = placement.records[0]
    assert isinstance(record, EvaluatedSampleRecord)
    outcome = record.executions[0].outcome
    assert isinstance(outcome, CandidateJobCompleted)
    wrong_result = outcome.result.model_copy(
        update={
            "candidate": outcome.result.candidate.model_copy(
                update={"candidate_ordinal": 1}
            )
        }
    )
    wrong_execution = record.executions[0].model_copy(
        update={"outcome": outcome.model_copy(update={"result": wrong_result})}
    )

    with pytest.raises(ValueError, match="completed candidate result"):
        validate_sample_record_graph(
            record.model_copy(update={"executions": (wrong_execution,)}),
            slot=batch_request.inputs[0].slot,
            sample=batch_request.inputs[0].sample.metadata.identity,
            plan=batch_request.plan,
            runtime=batch_request.runtime,
            cache_namespace=batch_request.cache_namespace,
        )
    await execution_cache.close()
