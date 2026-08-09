from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping

import pytest
from dr_serialize import (
    Jsonable,
    build_identity_document,
    identity_document_hash,
)
from dr_store import CacheEntry, CacheHit, ObjectReference, derive_cache_key

from dr_code.caching import (
    EXECUTION_CACHE_NAMESPACE,
    EXECUTION_CACHE_RECORD_SCHEMA,
    CachedExecutionObservation,
    WindowedExecutionCache,
)
from dr_code.evaluation import (
    EvaluationRuntimeIdentity,
    HarnessExecutionFailure,
    StoredRecordReference,
)

pytestmark = pytest.mark.asyncio


class _Store:
    def __init__(self) -> None:
        self.records: dict[str, tuple[str, Jsonable]] = {}
        self.get_calls: list[tuple[tuple[str, ...], str]] = []
        self.put_calls: list[dict[str, CacheEntry]] = []
        self.fail_writes = 0
        self.write_started: list[asyncio.Event] = []
        self.write_finished: list[asyncio.Event] = []
        self.write_releases: list[asyncio.Event | None] = []

    def gate_next_write(self) -> tuple[asyncio.Event, asyncio.Event]:
        started = asyncio.Event()
        release = asyncio.Event()
        self.write_started.append(started)
        self.write_finished.append(asyncio.Event())
        self.write_releases.append(release)
        return started, release

    async def get_many(
        self,
        keys: Iterable[str],
        *,
        schema: str,
    ) -> dict[str, CacheHit | None]:
        requested = tuple(keys)
        self.get_calls.append((requested, schema))
        results: dict[str, CacheHit | None] = {}
        for key in requested:
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
        call_index = len(self.put_calls)
        copied = dict(entries)
        self.put_calls.append(copied)
        while len(self.write_started) <= call_index:
            self.write_started.append(asyncio.Event())
            self.write_finished.append(asyncio.Event())
            self.write_releases.append(None)
        self.write_started[call_index].set()
        release = self.write_releases[call_index]
        try:
            if release is not None:
                await release.wait()
            if self.fail_writes:
                self.fail_writes -= 1
                raise OSError("write unavailable")
            for key, entry in copied.items():
                self.records[key] = (entry.schema, entry.record)
            return {
                key: ObjectReference.for_record(entry.schema, entry.record)
                for key, entry in copied.items()
            }
        finally:
            self.write_finished[call_index].set()


def _runtime(name: str = "runtime") -> EvaluationRuntimeIdentity:
    return EvaluationRuntimeIdentity(
        document=build_identity_document(
            schema="tests/runtime",
            schema_version=1,
            payload={"name": name},
        )
    )


def _observation(message: str = "observed") -> CachedExecutionObservation:
    return CachedExecutionObservation(
        source_record=StoredRecordReference(
            reference=ObjectReference.for_record(
                "dr-code/sample-evaluation-record-v1",
                {"fixture": "source"},
            ),
            schema_version=1,
        ),
        outcome=HarnessExecutionFailure(
            failure_type="FixtureFailure",
            message=message,
            execution_outcome=None,
            attribution=None,
            measurements=None,
        ),
    )


def _cache(
    store: _Store,
    *,
    resident: int = 4,
    pending: int = 4,
) -> WindowedExecutionCache:
    return WindowedExecutionCache(
        store,
        runtime=_runtime(),
        max_resident_entries=resident,
        max_pending_checkpoint_entries=pending,
    )


async def test_close_persists_and_restart_prefetch_restores_observation() -> (
    None
):
    store = _Store()
    first = _cache(store)
    await first.put("request", _observation("persisted"))
    await first.close()

    second = _cache(store)
    await second.prefetch(("request",))

    assert second.get("request") == _observation("persisted")
    assert second.stats().prefetched_entries == 1
    await second.close()


async def test_prefetch_bounds_hits_and_explicit_misses_as_resident_state() -> (
    None
):
    store = _Store()
    cache = _cache(store, resident=2)

    with pytest.raises(ValueError, match="max_resident_entries"):
        await cache.prefetch(("one", "two", "three"))
    assert store.get_calls == []

    await cache.prefetch(("one", "two", "one"))
    assert cache.get("one") is None
    assert cache.get("two") is None
    with pytest.raises(ValueError, match="max_resident_entries"):
        await cache.put("three", _observation())
    cache.discard("one")
    await cache.put("three", _observation())
    assert cache.get("three") == _observation()
    await cache.close()


async def test_one_in_flight_and_one_pending_batch_remain_bounded() -> None:
    store = _Store()
    started, release = store.gate_next_write()
    cache = _cache(store, resident=3, pending=1)

    await cache.put("one", _observation("one"))
    await started.wait()
    await cache.put("two", _observation("two"))
    await cache.put("three", _observation("three"))

    stats = cache.stats()
    assert stats.in_flight is True
    assert stats.dirty_entries == 2
    release.set()
    await cache.close()

    assert [len(batch) for batch in store.put_calls] == [1, 1]
    assert cache.stats().checkpoint_batches == 2
    persisted_messages = {
        entry.record["outcome"]["message"]
        for batch in store.put_calls
        for entry in batch.values()
    }
    assert persisted_messages == {"one", "two"}


async def test_failed_checkpoint_drops_retry_state_without_failing_cache() -> (
    None
):
    store = _Store()
    store.fail_writes = 1
    started, release = store.gate_next_write()
    cache = _cache(store, pending=1)

    await cache.put("request", _observation())
    await started.wait()
    release.set()
    await store.write_finished[0].wait()

    stats = cache.stats()
    assert stats.checkpoint_batches == 1
    assert stats.checkpoint_failures == 1
    assert stats.dirty_entries == 0
    assert cache.get("request") == _observation()
    await cache.close()
    assert len(store.put_calls) == 1


async def test_persistent_namespace_schema_keys_and_record_wire_are_golden() -> (
    None
):
    store = _Store()
    cache = _cache(store)
    observation = _observation()
    await cache.put("opaque-request", observation)
    await cache.close()

    expected_key = derive_cache_key(
        EXECUTION_CACHE_NAMESPACE,
        {
            "request_key": "opaque-request",
            "runtime_identity": str(
                identity_document_hash(_runtime().document)
            ),
        },
    )
    assert store.put_calls == [
        {
            expected_key: CacheEntry(
                schema=EXECUTION_CACHE_RECORD_SCHEMA,
                record=observation.model_dump(mode="json"),
            )
        }
    ]
    assert observation.model_dump(mode="json").keys() == {
        "schema_version",
        "source_record",
        "outcome",
    }
    assert observation.model_dump(mode="json")["source_record"].keys() == {
        "kind",
        "reference",
        "schema_version",
    }
    assert observation.model_dump(mode="json")["source_record"][
        "reference"
    ].keys() == {"schema", "content_hash"}
