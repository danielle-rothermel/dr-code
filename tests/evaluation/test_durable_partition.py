from __future__ import annotations

import asyncio
from threading import Event

import pytest
from dr_exec import CancelledOutcome, CompletedExecution, ExecutionJob

from _executor_stubs import (
    CountingExecutor,
    completed_execution,
    importable_json_executor,
)
from dr_code.evaluation import AttemptCompleteness
from dr_code.evaluation import _batch
from dr_code.evaluation._batch import _evaluate_durable_partition_assembly

from ._batch_builders import (
    BatchStore,
    MemoryPlacement,
    cache,
    frozen_input,
    request,
)

pytestmark = pytest.mark.asyncio


async def test_durable_partition_runs_serially_without_a_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_pool(*args: object, **kwargs: object) -> object:
        raise AssertionError("durable partition must not construct a pool")

    monkeypatch.setattr(_batch, "ExecutionPool", reject_pool)
    ordinary = request()
    frozen = frozen_input(0, ordinary.inputs[0].slot)
    batch_request = request(inputs=(frozen,))
    execution_cache = cache(BatchStore())
    executor = CountingExecutor(importable_json_executor())

    result = await _evaluate_durable_partition_assembly(
        batch_request,
        executor=executor,
        execution_cache=execution_cache,
        placement_sink=MemoryPlacement(),
    )

    assert result.completeness is AttemptCompleteness.COMPLETE
    assert executor.call_count == 1
    await execution_cache.close()


class _GateCancelToken:
    def __init__(self) -> None:
        self.requested = Event()

    def cancel(self) -> None:
        self.requested.set()


class _CancellationGatedExecutor:
    def __init__(self) -> None:
        self.started = Event()
        self.cleaned = Event()

    def run_blocking(
        self,
        job: ExecutionJob,
        /,
        *,
        cancellation: object = None,
    ) -> CompletedExecution:
        assert isinstance(cancellation, _GateCancelToken)
        self.started.set()
        assert cancellation.requested.wait(timeout=5)
        self.cleaned.set()
        return completed_execution(job, outcome=CancelledOutcome())

    def run(
        self,
        job: ExecutionJob,
        /,
        *,
        cancellation: object = None,
    ) -> CompletedExecution:
        return self.run_blocking(job, cancellation=cancellation)


async def test_durable_cancellation_requests_and_settles_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_batch, "CancelToken", _GateCancelToken)
    execution_cache = cache(BatchStore(), resident=1)
    executor = _CancellationGatedExecutor()
    try:
        running = asyncio.create_task(
            _evaluate_durable_partition_assembly(
                request(projections=()),
                executor=executor,
                execution_cache=execution_cache,
                placement_sink=MemoryPlacement(),
            )
        )

        assert await asyncio.wait_for(
            asyncio.to_thread(executor.started.wait), timeout=5
        )
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

        assert executor.cleaned.is_set()
        await execution_cache.prefetch(("post-cancellation-probe",))
        execution_cache.discard("post-cancellation-probe")
    finally:
        await execution_cache.close()
