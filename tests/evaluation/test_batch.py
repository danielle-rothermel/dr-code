from __future__ import annotations

import pytest
from dr_exec import AutoPoolCapacity, ExecutionPoolConfig

from _executor_stubs import CountingExecutor, importable_json_executor
from dr_code.evaluation import (
    AttemptCompleteness,
    AttemptValidity,
    EvaluatedSampleRecord,
)
from dr_code.evaluation import _batch
from dr_code.evaluation._batch import _evaluate_batch_assembly

from ._batch_builders import (
    BatchStore,
    MemoryPlacement,
    cache,
    frozen_input,
    request,
)

pytestmark = pytest.mark.asyncio


async def test_batch_uses_one_pool_and_preserves_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_pool = _batch.ExecutionPool
    pool_count = 0

    class CountingPool(real_pool):
        def __init__(self, *args: object, **kwargs: object) -> None:
            nonlocal pool_count
            pool_count += 1
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_batch, "ExecutionPool", CountingPool)
    batch_request = request(3)
    store = BatchStore()
    execution_cache = cache(store, resident=1)
    executor = CountingExecutor(importable_json_executor())
    placement = MemoryPlacement()

    result = await _evaluate_batch_assembly(
        batch_request,
        executor=executor,
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        placement_sink=placement,
    )

    assert pool_count == 1
    assert executor.call_count == 3
    assert result.completeness is AttemptCompleteness.COMPLETE
    assert result.validity is AttemptValidity.VALID
    assert [
        record.sample.identity.sample_id for record in placement.records
    ] == [
        "sample-0",
        "sample-1",
        "sample-2",
    ]
    assert all(
        isinstance(record, EvaluatedSampleRecord)
        for record in placement.records
    )
    assert result.aggregation is not None
    assert result.score is not None
    await execution_cache.close()


async def test_frozen_candidates_bypass_preprocessing_and_still_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = request()
    frozen = frozen_input(0, base.inputs[0].slot)
    batch_request = request(inputs=(frozen,))

    class RejectingRunner:
        def run(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("frozen candidates must bypass preprocessing")

    monkeypatch.setattr(
        _batch,
        "bind_preprocessing",
        lambda definition: RejectingRunner(),
    )
    store = BatchStore()
    execution_cache = cache(store)
    result = await _evaluate_batch_assembly(
        batch_request,
        executor=importable_json_executor(),
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        placement_sink=MemoryPlacement(),
    )
    assert result.completeness is AttemptCompleteness.COMPLETE
    await execution_cache.close()


async def test_cancellation_propagates_without_an_assembly() -> None:
    import asyncio

    from dr_exec import CancelledOutcome

    from _executor_stubs import scripted_executor

    batch_request = request()
    execution_cache = cache(BatchStore())
    try:
        with pytest.raises(asyncio.CancelledError):
            await _evaluate_batch_assembly(
                batch_request,
                executor=scripted_executor(outcome=CancelledOutcome()),
                execution_cache=execution_cache,
                pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
                placement_sink=MemoryPlacement(),
            )
    finally:
        await execution_cache.close()
