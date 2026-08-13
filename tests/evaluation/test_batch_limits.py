from __future__ import annotations

import pytest
from dr_exec import AutoPoolCapacity, ExecutionPoolConfig
from pydantic import ValidationError

from _executor_stubs import CountingExecutor, importable_json_executor
from dr_code.evaluation import (
    AttemptCompleteness,
    AttemptLimitKind,
    AttemptLimits,
    AttemptValidity,
    EvalBatchRequest,
    WindowLimits,
)
from dr_code.evaluation._batch import _evaluate_batch_assembly
from dr_code.trace import TextArtifact

from ._batch_builders import (
    BatchStore,
    MemoryPlacement,
    StoredMemoryPlacement,
    cache,
    request,
)

pytestmark = pytest.mark.asyncio


def _with_two_candidates(
    batch_request: EvalBatchRequest,
) -> EvalBatchRequest:
    selected_sample = batch_request.inputs[0].data.sample.model_copy(
        update={
            "raw_input": TextArtifact(
                text=(
                    "```python\ndef observed_load_count(_x): return 1\n```\n"
                    "```python\ndef observed_load_count(_x): return 2\n```"
                )
            )
        }
    )
    input_item = batch_request.inputs[0]
    return batch_request.model_copy(
        update={
            "inputs": (
                input_item.model_copy(
                    update={
                        "data": input_item.data.model_copy(
                            update={"sample": selected_sample}
                        )
                    }
                ),
            )
        }
    )


async def test_cache_hits_do_not_consume_admitted_job_limit() -> None:
    batch_request = request(
        attempt_limits=AttemptLimits(
            max_slots=1,
            max_materialized_candidates=2,
            max_admitted_jobs=1,
            max_retained_evidence_bytes=10_000_000,
            max_projection_rows=10,
        )
    )
    store = BatchStore()
    execution_cache = cache(store)
    executor = CountingExecutor(importable_json_executor())
    await _evaluate_batch_assembly(
        batch_request,
        executor=executor,
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        placement_sink=StoredMemoryPlacement(),
    )
    await execution_cache.close()
    execution_cache = cache(store)

    second = await _evaluate_batch_assembly(
        batch_request,
        executor=executor,
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        placement_sink=StoredMemoryPlacement(),
    )

    assert executor.call_count == 1
    assert second.completeness is AttemptCompleteness.COMPLETE
    assert second.limit_exhaustion is None
    await execution_cache.close()


async def test_every_materialized_candidate_consumes_materialized_limit() -> (
    None
):
    batch_request = request(
        attempt_limits=AttemptLimits(
            max_slots=1,
            max_materialized_candidates=1,
            max_admitted_jobs=4,
            max_retained_evidence_bytes=10_000_000,
            max_projection_rows=20,
        )
    )
    batch_request = _with_two_candidates(batch_request)
    execution_cache = cache(BatchStore())
    executor = CountingExecutor(importable_json_executor())

    placement = MemoryPlacement()
    result = await _evaluate_batch_assembly(
        batch_request,
        executor=executor,
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        placement_sink=placement,
    )

    assert result.completeness is AttemptCompleteness.PARTIAL
    assert result.validity is AttemptValidity.INVALID
    assert placement.records == []
    assert result.limit_exhaustion is not None
    assert (
        result.limit_exhaustion.limit
        is AttemptLimitKind.MATERIALIZED_CANDIDATES
    )
    assert result.limit_exhaustion.observed == 2
    assert executor.call_count == 0
    await execution_cache.close()


async def test_multi_candidate_cache_windows_release_resident_capacity() -> (
    None
):
    batch_request = _with_two_candidates(request(projections=()))
    execution_cache = cache(BatchStore(), resident=1)
    placement = StoredMemoryPlacement()

    result = await _evaluate_batch_assembly(
        batch_request,
        executor=importable_json_executor(),
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        placement_sink=placement,
    )

    assert result.completeness is AttemptCompleteness.COMPLETE
    assert len(placement.records) == 1
    await execution_cache.prefetch(("post-placement-probe",))
    execution_cache.discard("post-placement-probe")
    await execution_cache.close()


async def test_admission_exhaustion_after_earlier_window_releases_keys() -> (
    None
):
    batch_request = _with_two_candidates(
        request(
            projections=(),
            attempt_limits=AttemptLimits(
                max_slots=1,
                max_materialized_candidates=2,
                max_admitted_jobs=1,
                max_retained_evidence_bytes=10_000_000,
                max_projection_rows=10,
            ),
        )
    )
    execution_cache = cache(BatchStore(), resident=1)

    result = await _evaluate_batch_assembly(
        batch_request,
        executor=importable_json_executor(),
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        placement_sink=MemoryPlacement(),
    )

    assert result.limit_exhaustion is not None
    assert result.limit_exhaustion.limit is AttemptLimitKind.ADMITTED_JOBS
    await execution_cache.prefetch(("post-exhaustion-probe",))
    execution_cache.discard("post-exhaustion-probe")
    await execution_cache.close()


async def test_bundle_local_records_are_not_published_to_persistent_cache() -> (
    None
):
    store = BatchStore()
    execution_cache = cache(store, resident=1)
    await _evaluate_batch_assembly(
        request(projections=()),
        executor=importable_json_executor(),
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        placement_sink=MemoryPlacement(),
    )
    await execution_cache.close()

    assert store.records == {}


async def test_retained_evidence_exhaustion_preserves_only_completed_prefix() -> (
    None
):
    batch_request = request(
        2,
        attempt_limits=AttemptLimits(
            max_slots=2,
            max_materialized_candidates=4,
            max_admitted_jobs=4,
            max_retained_evidence_bytes=1,
            max_projection_rows=20,
        ),
    )
    execution_cache = cache(BatchStore())

    placement = MemoryPlacement()
    result = await _evaluate_batch_assembly(
        batch_request,
        executor=importable_json_executor(),
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        placement_sink=placement,
    )

    assert placement.records == []
    assert result.limit_exhaustion is not None
    assert (
        result.limit_exhaustion.limit
        is AttemptLimitKind.RETAINED_EVIDENCE_BYTES
    )
    assert result.limit_exhaustion.observed > 1
    await execution_cache.close()


async def test_admission_exhaustion_publishes_every_completed_sample() -> None:
    batch_request = request(
        3,
        attempt_limits=AttemptLimits(
            max_slots=3,
            max_materialized_candidates=6,
            max_admitted_jobs=2,
            max_retained_evidence_bytes=10_000_000,
            max_projection_rows=30,
        ),
    )
    execution_cache = cache(BatchStore())

    placement = MemoryPlacement()
    result = await _evaluate_batch_assembly(
        batch_request,
        executor=importable_json_executor(),
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        placement_sink=placement,
    )

    assert result.limit_exhaustion is not None
    assert result.limit_exhaustion.limit is AttemptLimitKind.ADMITTED_JOBS
    assert [
        record.sample.identity.sample_id for record in placement.records
    ] == [
        "sample-0",
        "sample-1",
    ]
    assert result.completeness is AttemptCompleteness.PARTIAL
    await execution_cache.close()


async def test_known_request_and_window_limit_violations_fail_model_validation() -> (
    None
):
    with pytest.raises(ValidationError, match="max_slots"):
        request(
            2,
            attempt_limits=AttemptLimits(
                max_slots=1,
                max_materialized_candidates=4,
                max_admitted_jobs=4,
                max_retained_evidence_bytes=10_000_000,
                max_projection_rows=20,
            ),
        )
    with pytest.raises(ValidationError, match="max_cache_keys"):
        request(
            window_limits=WindowLimits(
                max_preprocessing_slots=1,
                max_cache_keys=3,
                max_admitted_jobs=1,
                max_record_assemblies=1,
                max_projection_rows=5,
            ),
            attempt_limits=AttemptLimits(
                max_slots=1,
                max_materialized_candidates=2,
                max_admitted_jobs=2,
                max_retained_evidence_bytes=10_000_000,
                max_projection_rows=10,
            ),
        )
