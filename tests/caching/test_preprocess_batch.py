from __future__ import annotations

import threading

import pytest
from dr_exec import (
    ExecutionPoolConfig,
    FixedPoolCapacity,
    ImportableJsonExecutor,
)
from dr_store import MemoryBackend, ObjectStore, RecordCache

from dr_code.caching.preprocess_batch import (
    default_preprocess_batch_limits,
    preprocess_batch,
)
from dr_code.preprocessing import EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION

pytestmark = pytest.mark.asyncio

_FENCED = "Here is the code:\n```python\ndef f(x):\n    return x + 1\n```\n"


async def test_preprocess_batch_runs_distinct_texts() -> None:
    cache = RecordCache(ObjectStore(MemoryBackend()))
    results = await preprocess_batch(
        [_FENCED],
        definition=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
        store=cache,
        pool_config=ExecutionPoolConfig(
            capacity=FixedPoolCapacity(max_active_jobs=2)
        ),
        limits=default_preprocess_batch_limits(worker_count=2),
    )
    assert len(results) == 1
    assert _FENCED in results


async def test_preprocess_batch_reuses_cache_hits() -> None:
    cache = RecordCache(ObjectStore(MemoryBackend()))
    limits = default_preprocess_batch_limits(worker_count=2)
    pool_config = ExecutionPoolConfig(
        capacity=FixedPoolCapacity(max_active_jobs=2)
    )
    active = 0
    max_active = 0
    lock = threading.Lock()
    original_run = ImportableJsonExecutor.run

    def tracked_run(self, job, /, *, cancellation=None):  # noqa: ANN001
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            return original_run(self, job, cancellation=cancellation)
        finally:
            with lock:
                active -= 1

    ImportableJsonExecutor.run = tracked_run  # ty: ignore[method-assign]
    try:
        first = await preprocess_batch(
            [_FENCED],
            definition=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
            store=cache,
            pool_config=pool_config,
            limits=limits,
        )
        second = await preprocess_batch(
            [_FENCED],
            definition=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
            store=cache,
            pool_config=pool_config,
            limits=limits,
        )
    finally:
        ImportableJsonExecutor.run = original_run  # ty: ignore[method-assign]

    assert first == second
    assert max_active <= 2


async def test_preprocess_batch_respects_worker_limit() -> None:
    cache = RecordCache(ObjectStore(MemoryBackend()))
    sources = [
        f"```python\ndef f_{index}(x):\n    return x + {index}\n```"
        for index in range(8)
    ]
    active = 0
    max_active = 0
    lock = threading.Lock()
    original_run = ImportableJsonExecutor.run

    def slow_run(self, job, /, *, cancellation=None):  # noqa: ANN001
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            import time

            time.sleep(0.05)
            return original_run(self, job, cancellation=cancellation)
        finally:
            with lock:
                active -= 1

    ImportableJsonExecutor.run = slow_run  # ty: ignore[method-assign]
    try:
        await preprocess_batch(
            sources,
            definition=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
            store=cache,
            pool_config=ExecutionPoolConfig(
                capacity=FixedPoolCapacity(max_active_jobs=2)
            ),
            limits=default_preprocess_batch_limits(worker_count=2),
        )
    finally:
        ImportableJsonExecutor.run = original_run  # ty: ignore[method-assign]

    assert max_active <= 2
    assert max_active >= 2


async def test_preprocess_batch_discards_resident_entries_between_windows() -> (
    None
):
    cache = RecordCache(ObjectStore(MemoryBackend()))
    sources = [
        f"```python\ndef f_{index}(x):\n    return x + {index}\n```"
        for index in range(9)
    ]
    results = await preprocess_batch(
        sources,
        definition=EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
        store=cache,
        pool_config=ExecutionPoolConfig(
            capacity=FixedPoolCapacity(max_active_jobs=2)
        ),
        limits=default_preprocess_batch_limits(worker_count=2),
    )

    assert len(results) == len(sources)
    assert set(results) == set(sources)
