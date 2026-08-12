from __future__ import annotations

import pytest
from dr_exec import AutoPoolCapacity, ExecutionPoolConfig

from _executor_stubs import importable_json_executor
from dr_code.evaluation import (
    AggregationResultProjectionRow,
    BundleRecordReference,
    EvaluationSampleProjectionRow,
    MaterializedCandidateProjectionRow,
    MetricRecordProjectionRow,
    ProjectionKind,
    ScoreProjectionRow,
)
from dr_code.evaluation._batch import _evaluate_batch_assembly

from ._batch_builders import BatchStore, MemoryPlacement, cache, request

pytestmark = pytest.mark.asyncio


def _reference() -> BundleRecordReference:
    return BundleRecordReference(
        artifact_name="sample-records-00000000.jsonl",
        record_index=0,
        record_sha256="a" * 64,
        schema="dr-code/sample-evaluation-record-v1",
        schema_version=1,
    )


async def test_all_standard_projection_drafts_are_compact_and_ordered() -> (
    None
):
    batch_request = request()
    execution_cache = cache(BatchStore())
    placement = MemoryPlacement()
    assembly = await _evaluate_batch_assembly(
        batch_request,
        executor=importable_json_executor(),
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        placement_sink=placement,
    )

    assert len(placement.records) == 1
    assert assembly.aggregation is not None
    assert assembly.score is not None
    await execution_cache.close()


async def test_public_projection_rows_pin_the_five_wire_discriminators() -> (
    None
):
    batch_request = request()
    execution_cache = cache(BatchStore())
    placement = MemoryPlacement()
    assembly = await _evaluate_batch_assembly(
        batch_request,
        executor=importable_json_executor(),
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        placement_sink=placement,
    )
    record = placement.records[0]
    reference = _reference()
    candidate = record.candidates[0]  # type: ignore[union-attr]
    metric = record.metrics[0]  # type: ignore[union-attr]
    assert assembly.aggregation is not None
    assert assembly.score is not None

    rows = (
        EvaluationSampleProjectionRow(
            source_attempt=batch_request.attempt,
            slot=record.slot,
            sample=record.sample,
            status=record.status,
            record=reference,
        ),
        MaterializedCandidateProjectionRow(
            source_attempt=batch_request.attempt,
            candidate=candidate.identity,
            source_sha256=candidate.source_sha256,
            sample_record=reference,
        ),
        MetricRecordProjectionRow(
            source_attempt=batch_request.attempt,
            candidate=candidate.identity,
            question=metric.identity.question,
            status=metric.status,
            values=metric.values,  # type: ignore[union-attr]
            sample_record=reference,
        ),
        AggregationResultProjectionRow(
            source_attempt=batch_request.attempt,
            policy=batch_request.plan.aggregation,
            result=assembly.aggregation.result,
        ),
        ScoreProjectionRow(
            source_attempt=batch_request.attempt,
            score=assembly.score.score,
        ),
    )

    assert [row.model_dump(mode="json")["kind"] for row in rows] == [
        kind.value for kind in ProjectionKind
    ]
    assert all(
        "raw_input" not in row.model_dump_json()
        and "trace" not in row.model_dump_json()
        for row in rows
    )
    await execution_cache.close()
