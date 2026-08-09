from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Mapping
from threading import Lock

import pytest
from dr_exec import ExecutionJob, FakeExecutor
from dr_serialize import IdentityDocument, Jsonable
from dr_store import CacheEntry, CacheHit, ObjectReference

from _executor_stubs import completed_execution
from dr_code.caching import CheckpointedExecutionCache
from dr_code.metrics.engine.execution import (
    ExecutionOutcome,
    ExecutionRequest,
    run_requests,
)

_WATCHDOG_SECONDS = 5.0
pytestmark = pytest.mark.asyncio


def _runtime_identity(name: str = "cpython-test") -> IdentityDocument:
    return IdentityDocument(
        schema="tests/runtime",
        schema_version=1,
        payload={"name": name},
    )


def _outcome(stdout: str = "ok") -> ExecutionOutcome:
    return ExecutionOutcome(returncode=0, stdout=stdout, stderr="")


class _BatchStore:
    def __init__(self) -> None:
        self.records: dict[str, tuple[str, Jsonable]] = {}
        self.get_many_calls: list[tuple[tuple[str, ...], str]] = []
        self.get_many_results: list[dict[str, CacheHit | None]] = []
        self.put_many_calls: list[dict[str, tuple[str, Jsonable]]] = []
        self.fail_reads = 0
        self.fail_writes = 0
        self._write_gates: list[tuple[asyncio.Event, asyncio.Event]] = []
        self._write_finished: list[asyncio.Event] = []
        self._lock = Lock()

    def gate_write(self) -> tuple[asyncio.Event, asyncio.Event]:
        started = asyncio.Event()
        release = asyncio.Event()
        with self._lock:
            self._write_gates.append((started, release))
        return started, release

    def write_finished(self, call_index: int) -> asyncio.Event:
        with self._lock:
            while len(self._write_finished) <= call_index:
                self._write_finished.append(asyncio.Event())
            return self._write_finished[call_index]

    async def get_many(
        self,
        keys: Iterable[str],
        *,
        schema: str,
    ) -> dict[str, CacheHit | None]:
        requested = tuple(dict.fromkeys(keys))
        self.get_many_calls.append((requested, schema))
        if self.fail_reads:
            self.fail_reads -= 1
            raise OSError("read unavailable")
        results = {
            key: (
                CacheHit(record=stored[1])
                if (stored := self.records.get(key)) is not None
                and stored[0] == schema
                else None
            )
            for key in requested
        }
        self.get_many_results.append(results)
        return results

    async def put_many(
        self,
        entries: Mapping[str, CacheEntry],
    ) -> dict[str, ObjectReference]:
        with self._lock:
            call_index = len(self.put_many_calls)
            copied = {
                key: (entry.schema, entry.record)
                for key, entry in entries.items()
            }
            self.put_many_calls.append(copied)
            while len(self._write_finished) <= call_index:
                self._write_finished.append(asyncio.Event())
            finished = self._write_finished[call_index]
            gate = (
                self._write_gates[call_index]
                if call_index < len(self._write_gates)
                else None
            )
        try:
            if gate is not None:
                started, release = gate
                started.set()
                await asyncio.wait_for(
                    release.wait(), timeout=_WATCHDOG_SECONDS
                )
            if self.fail_writes:
                self.fail_writes -= 1
                raise OSError("write unavailable")
            references: dict[str, ObjectReference] = {}
            for key, entry in copied.items():
                self.records.setdefault(key, entry)
                stored_schema, stored_record = self.records[key]
                references[key] = ObjectReference.for_record(
                    stored_schema,
                    stored_record,
                )
            return references
        finally:
            finished.set()


def _cache(
    store: _BatchStore,
    *,
    runtime: str = "cpython-test",
    checkpoint_entry_count: int = 100,
) -> CheckpointedExecutionCache:
    return CheckpointedExecutionCache(
        store,
        runtime_identity=_runtime_identity(runtime),
        checkpoint_entry_count=checkpoint_entry_count,
    )


async def test_restart_prefetch_restores_checkpointed_outcome() -> None:
    store = _BatchStore()
    first = _cache(store)
    await first.put("request", _outcome("persisted"))
    await first.close()

    second = _cache(store)
    await second.prefetch(("request",))

    assert second.get("request") == _outcome("persisted")
    assert second.stats().prefetched_entries == 1
    await second.close()


async def test_prefetch_only_requests_previously_unseen_keys() -> None:
    store = _BatchStore()
    cache = _cache(store)

    await cache.prefetch(("one", "two", "one"))
    await cache.prefetch(("two", "three"))

    assert [len(keys) for keys, _ in store.get_many_calls] == [2, 1]
    assert set(store.get_many_calls[0][0]).isdisjoint(
        store.get_many_calls[1][0]
    )
    await cache.close()


async def test_hot_get_and_put_do_not_read_persistence() -> None:
    store = _BatchStore()
    cache = _cache(store)
    await cache.prefetch(("request",))

    persistent_key = store.get_many_calls[0][0][0]
    assert store.get_many_results == [{persistent_key: None}]
    assert cache.get("request") is None
    await cache.put("request", _outcome())
    assert cache.get("request") == _outcome()
    assert cache.get("request") == _outcome()

    assert len(store.get_many_calls) == 1
    assert store.put_many_calls == []
    assert cache.stats().memory_hits == 2
    assert cache.stats().memory_misses == 1
    await cache.close()


async def test_runtime_identity_separates_persistent_keys() -> None:
    store = _BatchStore()
    first = _cache(store, runtime="runtime-one")
    second = _cache(store, runtime="runtime-two")

    await first.prefetch(("same-request",))
    await second.prefetch(("same-request",))

    assert store.get_many_calls[0][0] != store.get_many_calls[1][0]
    await first.close()
    await second.close()


async def test_threshold_automatically_starts_checkpoint() -> None:
    store = _BatchStore()
    started, release = store.gate_write()
    cache = _cache(store, checkpoint_entry_count=2)

    await cache.put("one", _outcome("one"))
    assert not started.is_set()
    await cache.put("two", _outcome("two"))
    assert started.is_set()

    stats = cache.stats()
    assert stats.in_flight
    assert stats.dirty_entries == 0
    release.set()
    await asyncio.wait_for(
        store.write_finished(0).wait(), timeout=_WATCHDOG_SECONDS
    )
    await cache.close()

    assert len(store.put_many_calls) == 1
    assert cache.stats().checkpoint_batches == 1
    assert cache.stats().checkpoint_entries == 2


async def test_checkpoint_starts_before_second_execution() -> None:
    store = _BatchStore()
    first_started, first_release = store.gate_write()
    cache = _cache(store, checkpoint_entry_count=1)
    second_invoked = asyncio.Event()
    invocation_count = 0

    def respond(job: ExecutionJob, cancellation: object):
        nonlocal invocation_count
        del cancellation
        invocation_count += 1
        if invocation_count == 2:
            second_invoked.set()
            assert first_started.is_set()
        return completed_execution(job, stdout="[]")

    requests = (
        ExecutionRequest(
            source="def dr_exec_main(request, emit):\n    pass\n",
            input_json='{"request": 1}',
            timeout_seconds=1.0,
            computation_id="first",
        ),
        ExecutionRequest(
            source="def dr_exec_main(request, emit):\n    pass\n",
            input_json='{"request": 2}',
            timeout_seconds=1.0,
            computation_id="second",
        ),
    )
    execution = asyncio.create_task(
        run_requests(
            requests,
            executor=FakeExecutor(responder=respond),
            cache=cache,
        )
    )

    try:
        await asyncio.wait_for(
            second_invoked.wait(), timeout=_WATCHDOG_SECONDS
        )
        assert not store.write_finished(0).is_set()
    finally:
        first_release.set()

    outcomes = await asyncio.wait_for(execution, timeout=_WATCHDOG_SECONDS)
    await cache.close()

    assert tuple(outcomes) == requests


async def test_writes_coalesce_while_one_checkpoint_is_blocked() -> None:
    store = _BatchStore()
    first_started, first_release = store.gate_write()
    second_started, second_release = store.gate_write()
    second_release.set()
    cache = _cache(store, checkpoint_entry_count=1)

    first_put = asyncio.create_task(cache.put("one", _outcome("one")))
    await asyncio.wait_for(first_started.wait(), timeout=_WATCHDOG_SECONDS)
    await first_put
    second_put = asyncio.create_task(cache.put("two", _outcome("two")))
    third_put = asyncio.create_task(cache.put("three", _outcome("three")))
    puts_started = asyncio.Event()
    asyncio.get_running_loop().call_soon(puts_started.set)
    await puts_started.wait()
    assert cache.stats().dirty_entries == 2
    first_release.set()

    await asyncio.wait_for(second_started.wait(), timeout=_WATCHDOG_SECONDS)
    await asyncio.gather(second_put, third_put)
    await asyncio.wait_for(
        store.write_finished(1).wait(), timeout=_WATCHDOG_SECONDS
    )
    await cache.close()

    assert [len(entries) for entries in store.put_many_calls] == [1, 2]
    assert cache.stats().checkpoint_batches == 2
    assert cache.stats().checkpoint_entries == 3


async def test_failed_batch_is_retained_for_later_checkpoint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _BatchStore()
    store.fail_writes = 1
    cache = _cache(store)
    await cache.put("request", _outcome())

    with caplog.at_level(logging.WARNING):
        cache.checkpoint()
        await asyncio.wait_for(
            store.write_finished(0).wait(), timeout=_WATCHDOG_SECONDS
        )
        # This request is made after the failed store call. Whether the writer
        # has reacquired its state lock yet is immaterial: it must retain the
        # failed snapshot and make it the next eligible batch.
        cache.checkpoint()
        await asyncio.wait_for(
            store.write_finished(1).wait(), timeout=_WATCHDOG_SECONDS
        )
        await cache.close()

    stats = cache.stats()
    assert stats.dirty_entries == 0
    assert stats.checkpoint_batches == 2
    assert stats.checkpoint_entries == 1
    assert stats.checkpoint_failures == 1
    assert len(store.records) == 1
    assert any(
        "retaining entries" in record.message for record in caplog.records
    )


async def test_close_drains_a_final_checkpoint() -> None:
    store = _BatchStore()
    cache = _cache(store, checkpoint_entry_count=10)
    await cache.put("request", _outcome())

    assert store.put_many_calls == []
    await cache.close()

    assert len(store.put_many_calls) == 1
    assert cache.stats().dirty_entries == 0
    assert not cache.stats().in_flight


async def test_cancelled_close_waits_for_checkpoint_settlement() -> None:
    store = _BatchStore()
    started, release = store.gate_write()
    cache = _cache(store, checkpoint_entry_count=10)
    await cache.put("request", _outcome())

    closing = asyncio.create_task(cache.close())
    await asyncio.wait_for(started.wait(), timeout=_WATCHDOG_SECONDS)
    closing.cancel()
    cancellation_delivered = asyncio.Event()
    asyncio.get_running_loop().call_soon(cancellation_delivered.set)
    await cancellation_delivered.wait()

    assert not closing.done()
    assert cache.stats().in_flight
    assert cache.stats().dirty_entries == 0

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closing, timeout=_WATCHDOG_SECONDS)

    assert store.write_finished(0).is_set()
    assert cache.stats().dirty_entries == 0
    assert not cache.stats().in_flight
    await cache.close()


async def test_malformed_persisted_outcome_is_an_observable_miss(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _BatchStore()
    seed = _cache(store)
    await seed.prefetch(("request",))
    persistent_key = store.get_many_calls[0][0][0]
    schema = store.get_many_calls[0][1]
    await seed.close()
    store.records[persistent_key] = (
        schema,
        {
            "returncode": "0",
            "stdout": "not strict",
            "stderr": "",
        },
    )

    cache = _cache(store)
    with caplog.at_level(logging.WARNING):
        await cache.prefetch(("request",))

    assert cache.get("request") is None
    assert any(
        "invalid execution cache entry" in record.message
        for record in caplog.records
    )
    await cache.close()


async def test_prefetch_failure_is_logged_and_does_not_escape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _BatchStore()
    store.fail_reads = 1
    cache = _cache(store)

    with caplog.at_level(logging.WARNING):
        await cache.prefetch(("request",))

    assert cache.get("request") is None
    assert any(
        "prefetch failed" in record.message for record in caplog.records
    )
    await cache.close()


@pytest.mark.parametrize("entry_count", [0, -1, True])
async def test_checkpoint_entry_count_must_be_positive(
    entry_count: int,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        CheckpointedExecutionCache(
            _BatchStore(),
            runtime_identity=_runtime_identity(),
            checkpoint_entry_count=entry_count,
        )
