from __future__ import annotations

import pytest
from dr_exec import ExecutionPoolConfig, SpawnAbsentOutcome

from _executor_stubs import (
    CountingExecutor,
    importable_json_executor,
    scripted_executor,
    timeout_executor,
)
from dr_code.evaluation._batch import _evaluate_batch_assembly

from ._batch_builders import (
    BatchStore,
    StoredMemoryPlacement,
    cache,
    request,
)

pytestmark = pytest.mark.asyncio


def _infrastructure_executor() -> object:
    """Return an executor whose outcome dr-code attributes to the executor."""

    return scripted_executor(
        outcome=SpawnAbsentOutcome(executable="/nonexistent/interpreter")
    )


async def _run(
    batch_request: object,
    *,
    executor: object,
    store: BatchStore,
) -> None:
    execution_cache = cache(store)
    await _evaluate_batch_assembly(
        batch_request,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(),
        placement_sink=StoredMemoryPlacement(),
    )
    await execution_cache.close()


async def test_executor_failure_is_never_persisted_and_re_executes() -> None:
    batch_request = request(projections=())
    store = BatchStore()
    executor = CountingExecutor(_infrastructure_executor())  # type: ignore[arg-type]

    await _run(batch_request, executor=executor, store=store)

    assert store.records == {}

    await _run(batch_request, executor=executor, store=store)

    assert executor.call_count == 2


async def test_candidate_failure_is_persisted_and_reused() -> None:
    batch_request = request(projections=())
    store = BatchStore()
    executor = CountingExecutor(timeout_executor())

    await _run(batch_request, executor=executor, store=store)

    assert len(store.records) == 1

    await _run(batch_request, executor=executor, store=store)

    assert executor.call_count == 1


async def test_completed_outcome_is_persisted_and_reused() -> None:
    batch_request = request(projections=())
    store = BatchStore()
    executor = CountingExecutor(importable_json_executor())

    await _run(batch_request, executor=executor, store=store)

    assert len(store.records) == 1

    await _run(batch_request, executor=executor, store=store)

    assert executor.call_count == 1


async def test_fresh_request_re_executes_despite_a_populated_cache() -> None:
    batch_request = request(projections=())
    store = BatchStore()
    executor = CountingExecutor(importable_json_executor())

    await _run(batch_request, executor=executor, store=store)

    assert len(store.records) == 1
    assert executor.call_count == 1

    fresh_request = batch_request.model_copy(update={"fresh": True})
    await _run(fresh_request, executor=executor, store=store)

    assert executor.call_count == 2


async def test_fresh_request_never_reads_the_persistent_cache() -> None:
    batch_request = request(projections=())
    store = BatchStore()

    await _run(
        batch_request,
        executor=importable_json_executor(),
        store=store,
    )
    read_calls = len(store.get_calls)

    await _run(
        batch_request.model_copy(update={"fresh": True}),
        executor=importable_json_executor(),
        store=store,
    )

    assert len(store.get_calls) == read_calls


async def test_fresh_outcome_is_the_one_offered_to_persistence() -> None:
    batch_request = request(projections=())
    store = BatchStore()

    await _run(
        batch_request,
        executor=_infrastructure_executor(),
        store=store,
    )

    assert store.records == {}

    await _run(
        batch_request.model_copy(update={"fresh": True}),
        executor=importable_json_executor(),
        store=store,
    )

    assert len(store.records) == 1
