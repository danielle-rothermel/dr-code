from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from threading import Event, Lock

import pytest
from dr_serialize import IdentityDocument, Jsonable
from dr_store import CacheEntry, CacheHit, ObjectReference

from dr_code.caching import CheckpointedExecutionCache
from dr_code.metrics.engine.execution import ExecutionOutcome

_WATCHDOG_SECONDS = 5.0


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
        self._write_gates: list[tuple[Event, Event]] = []
        self._write_finished: list[Event] = []
        self._lock = Lock()

    def gate_write(self) -> tuple[Event, Event]:
        started = Event()
        release = Event()
        with self._lock:
            self._write_gates.append((started, release))
        return started, release

    def write_finished(self, call_index: int) -> Event:
        with self._lock:
            while len(self._write_finished) <= call_index:
                self._write_finished.append(Event())
            return self._write_finished[call_index]

    def get_many(
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

    def put_many(
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
                self._write_finished.append(Event())
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
                if not release.wait(timeout=_WATCHDOG_SECONDS):
                    raise TimeoutError("test write gate was not released")
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


def test_restart_prefetch_restores_checkpointed_outcome() -> None:
    store = _BatchStore()
    first = _cache(store)
    first.put("request", _outcome("persisted"))
    first.close()

    second = _cache(store)
    second.prefetch(("request",))

    assert second.get("request") == _outcome("persisted")
    assert second.stats().prefetched_entries == 1
    second.close()


def test_prefetch_only_requests_previously_unseen_keys() -> None:
    store = _BatchStore()
    cache = _cache(store)

    cache.prefetch(("one", "two", "one"))
    cache.prefetch(("two", "three"))

    assert [len(keys) for keys, _ in store.get_many_calls] == [2, 1]
    assert set(store.get_many_calls[0][0]).isdisjoint(
        store.get_many_calls[1][0]
    )
    cache.close()


def test_hot_get_and_put_do_not_read_persistence() -> None:
    store = _BatchStore()
    cache = _cache(store)
    cache.prefetch(("request",))

    persistent_key = store.get_many_calls[0][0][0]
    assert store.get_many_results == [{persistent_key: None}]
    assert cache.get("request") is None
    cache.put("request", _outcome())
    assert cache.get("request") == _outcome()
    assert cache.get("request") == _outcome()

    assert len(store.get_many_calls) == 1
    assert store.put_many_calls == []
    assert cache.stats().memory_hits == 2
    assert cache.stats().memory_misses == 1
    cache.close()


def test_runtime_identity_separates_persistent_keys() -> None:
    store = _BatchStore()
    first = _cache(store, runtime="runtime-one")
    second = _cache(store, runtime="runtime-two")

    first.prefetch(("same-request",))
    second.prefetch(("same-request",))

    assert store.get_many_calls[0][0] != store.get_many_calls[1][0]
    first.close()
    second.close()


def test_threshold_automatically_starts_checkpoint() -> None:
    store = _BatchStore()
    started, release = store.gate_write()
    cache = _cache(store, checkpoint_entry_count=2)

    cache.put("one", _outcome("one"))
    assert not started.is_set()
    cache.put("two", _outcome("two"))
    assert started.wait(timeout=_WATCHDOG_SECONDS)

    stats = cache.stats()
    assert stats.in_flight
    assert stats.dirty_entries == 0
    release.set()
    assert store.write_finished(0).wait(timeout=_WATCHDOG_SECONDS)
    cache.close()

    assert len(store.put_many_calls) == 1
    assert cache.stats().checkpoint_batches == 1
    assert cache.stats().checkpoint_entries == 2


def test_writes_coalesce_while_one_checkpoint_is_blocked() -> None:
    store = _BatchStore()
    first_started, first_release = store.gate_write()
    second_started, second_release = store.gate_write()
    second_release.set()
    cache = _cache(store, checkpoint_entry_count=1)

    cache.put("one", _outcome("one"))
    assert first_started.wait(timeout=_WATCHDOG_SECONDS)
    cache.put("two", _outcome("two"))
    cache.put("three", _outcome("three"))
    first_release.set()

    assert second_started.wait(timeout=_WATCHDOG_SECONDS)
    assert store.write_finished(1).wait(timeout=_WATCHDOG_SECONDS)
    cache.close()

    assert [len(entries) for entries in store.put_many_calls] == [1, 2]
    assert cache.stats().checkpoint_batches == 2
    assert cache.stats().checkpoint_entries == 3


def test_failed_batch_is_retained_for_later_checkpoint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _BatchStore()
    store.fail_writes = 1
    cache = _cache(store)
    cache.put("request", _outcome())

    with caplog.at_level(logging.WARNING):
        cache.checkpoint()
        assert store.write_finished(0).wait(timeout=_WATCHDOG_SECONDS)
        # This request is made after the failed store call. Whether the writer
        # has reacquired its state lock yet is immaterial: it must retain the
        # failed snapshot and make it the next eligible batch.
        cache.checkpoint()
        assert store.write_finished(1).wait(timeout=_WATCHDOG_SECONDS)
        cache.close()

    stats = cache.stats()
    assert stats.dirty_entries == 0
    assert stats.checkpoint_batches == 2
    assert stats.checkpoint_entries == 1
    assert stats.checkpoint_failures == 1
    assert len(store.records) == 1
    assert any(
        "retaining entries" in record.message for record in caplog.records
    )


def test_close_drains_a_final_checkpoint() -> None:
    store = _BatchStore()
    cache = _cache(store, checkpoint_entry_count=10)
    cache.put("request", _outcome())

    assert store.put_many_calls == []
    cache.close()

    assert len(store.put_many_calls) == 1
    assert cache.stats().dirty_entries == 0
    assert not cache.stats().in_flight


def test_malformed_persisted_outcome_is_an_observable_miss(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _BatchStore()
    seed = _cache(store)
    seed.prefetch(("request",))
    persistent_key = store.get_many_calls[0][0][0]
    schema = store.get_many_calls[0][1]
    seed.close()
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
        cache.prefetch(("request",))

    assert cache.get("request") is None
    assert any(
        "invalid execution cache entry" in record.message
        for record in caplog.records
    )
    cache.close()


def test_prefetch_failure_is_logged_and_does_not_escape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _BatchStore()
    store.fail_reads = 1
    cache = _cache(store)

    with caplog.at_level(logging.WARNING):
        cache.prefetch(("request",))

    assert cache.get("request") is None
    assert any(
        "prefetch failed" in record.message for record in caplog.records
    )
    cache.close()


@pytest.mark.parametrize("entry_count", [0, -1, True])
def test_checkpoint_entry_count_must_be_positive(entry_count: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        CheckpointedExecutionCache(
            _BatchStore(),
            runtime_identity=_runtime_identity(),
            checkpoint_entry_count=entry_count,
        )
