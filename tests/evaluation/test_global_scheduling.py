from __future__ import annotations

import pytest
from dr_exec import ExecutionPoolConfig, FixedPoolCapacity

from _executor_stubs import (
    ConcurrencyTrackingExecutor,
    importable_json_executor,
)
from dr_code.evaluation import (
    AttemptCompleteness,
    AttemptLimits,
    EvaluationBatchRequest,
    EvaluationSlotIdentity,
    WindowLimits,
)
from dr_code.evaluation._batch import _evaluate_batch_assembly

from ._batch_builders import (
    TASK_ID,
    BatchStore,
    MemoryPlacement,
    cache,
    frozen_input,
    request,
)

pytestmark = pytest.mark.asyncio


def _multi_frozen_request(
    count: int,
    *,
    attempt_limits: AttemptLimits | None = None,
    window_limits: WindowLimits | None = None,
) -> EvaluationBatchRequest:
    base = request(
        count,
        attempt_limits=attempt_limits,
        window_limits=window_limits,
    )
    task_set = base.plan.task_set.coordinate
    repeat_plan = base.plan.repeat_plan.coordinate
    inputs = tuple(
        frozen_input(
            index,
            EvaluationSlotIdentity(
                task_set=task_set,
                repeat_plan=repeat_plan,
                task_id=TASK_ID,
                repeat_index=index,
            ),
        )
        for index in range(count)
    )
    return base.model_copy(update={"inputs": inputs})


async def test_global_scheduler_spreads_concurrency_across_samples() -> None:
    batch_request = _multi_frozen_request(
        24,
        attempt_limits=AttemptLimits(
            max_slots=24,
            max_materialized_candidates=24,
            max_admitted_jobs=24,
            max_retained_evidence_bytes=10_000_000,
            max_projection_rows=200,
        ),
        window_limits=WindowLimits(
            max_preprocessing_slots=1,
            max_cache_keys=24,
            max_admitted_jobs=8,
            max_record_assemblies=24,
            max_projection_rows=200,
        ),
    )
    execution_cache = cache(BatchStore(), resident=64)
    executor = ConcurrencyTrackingExecutor(
        importable_json_executor(),
        delay_seconds=0.05,
    )
    try:
        result = await _evaluate_batch_assembly(
            batch_request,
            executor=executor,
            execution_cache=execution_cache,
            pool_config=ExecutionPoolConfig(
                capacity=FixedPoolCapacity(max_active_jobs=8)
            ),
            placement_sink=MemoryPlacement(),
        )
    finally:
        await execution_cache.close()

    assert result.completeness is AttemptCompleteness.COMPLETE
    assert executor.max_active >= 8


async def test_global_scheduler_still_respects_admission_limit() -> None:
    batch_request = _multi_frozen_request(
        2,
        attempt_limits=AttemptLimits(
            max_slots=2,
            max_materialized_candidates=8,
            max_admitted_jobs=2,
            max_retained_evidence_bytes=10_000_000,
            max_projection_rows=20,
        ),
        window_limits=WindowLimits(
            max_preprocessing_slots=1,
            max_cache_keys=8,
            max_admitted_jobs=2,
            max_record_assemblies=2,
            max_projection_rows=20,
        ),
    )
    execution_cache = cache(BatchStore(), resident=8)
    executor = ConcurrencyTrackingExecutor(importable_json_executor())
    try:
        result = await _evaluate_batch_assembly(
            batch_request,
            executor=executor,
            execution_cache=execution_cache,
            pool_config=ExecutionPoolConfig(
                capacity=FixedPoolCapacity(max_active_jobs=2)
            ),
            placement_sink=MemoryPlacement(),
        )
    finally:
        await execution_cache.close()

    assert result.completeness is AttemptCompleteness.COMPLETE
    assert executor.max_active <= 2
