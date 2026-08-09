from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from dr_exec import ExecutionPoolConfig
from dr_store import ArtifactBundlePublication

from _executor_stubs import importable_json_executor
from dr_code.evaluation import evaluate_batch

from ._batch_builders import BatchStore, cache, request

pytestmark = pytest.mark.asyncio


class GatedStore(BatchStore):
    def __init__(self) -> None:
        super().__init__()
        self.started: asyncio.Queue[tuple[asyncio.Event, int]] = (
            asyncio.Queue()
        )

    async def get_many(self, keys, *, schema):  # type: ignore[no-untyped-def]
        requested = tuple(keys)
        release = asyncio.Event()
        await self.started.put((release, len(requested)))
        await release.wait()
        return await super().get_many(requested, schema=schema)


async def test_cache_and_job_windows_remain_bounded_across_larger_input(
    tmp_path: Path,
) -> None:
    store = GatedStore()
    batch_request = request(4)
    execution_cache = cache(store, resident=1)
    publication = ArtifactBundlePublication.allocate(
        tmp_path, prefix="resident"
    )
    running = asyncio.create_task(
        evaluate_batch(
            batch_request,
            executor=importable_json_executor(),
            execution_cache=execution_cache,
            object_store=None,
            publication=publication,
            pool_config=ExecutionPoolConfig(),
        )
    )

    observed_sizes: list[int] = []
    for expected_call_count in range(1, 5):
        release, size = await store.started.get()
        observed_sizes.append(size)
        assert len(store.get_calls) == expected_call_count - 1
        release.set()
    result = await running

    assert observed_sizes == [1, 1, 1, 1]
    assert len(result.attempt.members) == 4
    assert result.bundle_path is not None
    assert len(tuple(result.bundle_path.glob("sample-records-*.jsonl"))) == 4
    assert max(len(call) for call in store.get_calls) == 1
    assert execution_cache.stats().dirty_entries <= 1
    await execution_cache.close()
