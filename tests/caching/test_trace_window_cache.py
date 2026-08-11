from __future__ import annotations

from collections.abc import Iterable, Mapping

import pytest
from dr_serialize import Jsonable
from dr_store import CacheEntry, CacheHit, ObjectReference

from dr_code.caching.trace_window_cache import WindowedTraceCache
from dr_code.preprocessing import (
    EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION,
    bind_preprocessing,
)
from dr_code.trace import TextArtifact, serialize_trace

pytestmark = pytest.mark.asyncio


class _Store:
    def __init__(self) -> None:
        self.records: dict[str, tuple[str, Jsonable]] = {}
        self.put_calls: list[dict[str, CacheEntry]] = []

    async def get_many(
        self,
        keys: Iterable[str],
        *,
        schema: str,
    ) -> dict[str, CacheHit | None]:
        results: dict[str, CacheHit | None] = {}
        for key in keys:
            stored = self.records.get(key)
            results[key] = (
                CacheHit(record=stored[1])
                if stored is not None and stored[0] == schema
                else None
            )
        return results

    async def put_many(
        self,
        entries: Mapping[str, CacheEntry],
    ) -> dict[str, ObjectReference]:
        copied = dict(entries)
        self.put_calls.append(copied)
        for key, entry in copied.items():
            self.records[key] = (entry.schema, entry.record)
        return {key: ObjectReference(content_hash=key) for key in copied}


async def test_trace_cache_prefetch_and_checkpoint() -> None:
    store = _Store()
    runner = bind_preprocessing(EXHAUSTIVE_FUNCTION_CANDIDATES_DEFINITION)
    trace = serialize_trace(runner.run(TextArtifact(text="hello")))
    key = "cache-key"

    async with WindowedTraceCache(
        store,
        max_resident_entries=4,
        max_pending_checkpoint_entries=2,
    ) as cache:
        await cache.prefetch([key])
        assert cache.get(key) is None
        await cache.put(key, trace)
        await cache.close()

    assert store.put_calls
    assert key in store.records
