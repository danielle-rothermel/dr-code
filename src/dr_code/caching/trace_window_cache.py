from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from dr_store import CacheEntry

from dr_code.caching.execution_cache import BatchRecordStore
from dr_code.caching.trace_cache import TRACE_RECORD_SCHEMA
from dr_code.trace.serialization import SerializedTrace

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TraceCacheStats:
    prefetched_entries: int
    memory_hits: int
    memory_misses: int
    dirty_entries: int
    checkpoint_batches: int
    checkpoint_entries: int
    checkpoint_failures: int
    in_flight: bool


class WindowedTraceCache:
    """Bounded resident traces with bounded best-effort checkpointing."""

    def __init__(
        self,
        store: BatchRecordStore,
        *,
        max_resident_entries: int,
        max_pending_checkpoint_entries: int,
    ) -> None:
        _require_positive_int(max_resident_entries, "max_resident_entries")
        _require_positive_int(
            max_pending_checkpoint_entries,
            "max_pending_checkpoint_entries",
        )
        self._store = store
        self._max_resident_entries = max_resident_entries
        self._max_pending_checkpoint_entries = max_pending_checkpoint_entries
        self._resident: dict[str, SerializedTrace | None] = {}
        self._pending: dict[str, SerializedTrace] = {}
        self._in_flight_batch: dict[str, SerializedTrace] | None = None
        self._checkpoint_event = asyncio.Event()
        self._closing = False
        self._closed = False
        self._prefetched_entries = 0
        self._memory_hits = 0
        self._memory_misses = 0
        self._checkpoint_batches = 0
        self._checkpoint_entries = 0
        self._checkpoint_failures = 0
        self._writer = asyncio.create_task(
            self._write_checkpoints(),
            name="dr-code-windowed-trace-cache-writer",
        )

    async def __aenter__(self) -> WindowedTraceCache:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def prefetch(self, cache_keys: Iterable[str], /) -> None:
        self._require_open()
        keys = tuple(dict.fromkeys(cache_keys))
        unseen = tuple(key for key in keys if key not in self._resident)
        if len(self._resident) + len(unseen) > self._max_resident_entries:
            raise ValueError(
                "trace cache prefetch exceeds max_resident_entries"
            )
        if not unseen:
            return

        try:
            hits = await self._store.get_many(
                unseen, schema=TRACE_RECORD_SCHEMA
            )
        except Exception:
            _LOGGER.warning(
                "trace cache prefetch failed; treating entries as misses",
                exc_info=True,
            )
            hits = {}

        restored: dict[str, SerializedTrace | None] = {}
        for cache_key in unseen:
            hit = hits.get(cache_key)
            if hit is None:
                restored[cache_key] = None
                continue
            try:
                restored[cache_key] = SerializedTrace.model_validate(
                    hit.record
                )
            except Exception:
                _LOGGER.warning(
                    "invalid trace cache entry; treating it as a miss",
                    exc_info=True,
                )
                restored[cache_key] = None

        self._require_open()
        for key, trace in restored.items():
            if key in self._resident:
                continue
            self._resident[key] = trace
            if trace is not None:
                self._prefetched_entries += 1

    def get(self, cache_key: str, /) -> SerializedTrace | None:
        self._require_open()
        trace = self._resident.get(cache_key)
        if trace is None:
            self._memory_misses += 1
        else:
            self._memory_hits += 1
        return trace

    async def put(self, cache_key: str, trace: SerializedTrace, /) -> None:
        self._require_open()
        if (
            cache_key not in self._resident
            and len(self._resident) >= self._max_resident_entries
        ):
            raise ValueError("trace cache put exceeds max_resident_entries")
        self._resident[cache_key] = trace

        if cache_key in self._pending:
            self._pending[cache_key] = trace
        elif len(self._pending) < self._max_pending_checkpoint_entries:
            self._pending[cache_key] = trace
        if (
            len(self._pending) >= self._max_pending_checkpoint_entries
            and self._in_flight_batch is None
        ):
            self._checkpoint_event.set()

    def discard(self, cache_key: str, /) -> None:
        self._require_open()
        self._resident.pop(cache_key, None)

    def stats(self) -> TraceCacheStats:
        return TraceCacheStats(
            prefetched_entries=self._prefetched_entries,
            memory_hits=self._memory_hits,
            memory_misses=self._memory_misses,
            dirty_entries=len(self._pending)
            + (
                0
                if self._in_flight_batch is None
                else len(self._in_flight_batch)
            ),
            checkpoint_batches=self._checkpoint_batches,
            checkpoint_entries=self._checkpoint_entries,
            checkpoint_failures=self._checkpoint_failures,
            in_flight=self._in_flight_batch is not None,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closing = True
        self._checkpoint_event.set()
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await asyncio.shield(self._writer)
            except asyncio.CancelledError as error:
                if cancellation is None:
                    cancellation = error
                if not self._writer.done():
                    continue
                if self._writer.cancelled():
                    raise cancellation
                try:
                    self._writer.result()
                except BaseException as writer_error:
                    raise cancellation from writer_error
                raise cancellation
            if cancellation is not None:
                raise cancellation
            return

    def _require_open(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("trace cache is closed")

    async def _write_checkpoints(self) -> None:
        while True:
            await self._checkpoint_event.wait()
            self._checkpoint_event.clear()
            if self._pending:
                batch = self._pending
                self._pending = {}
                self._in_flight_batch = batch
                persisted = await self._persist_batch(batch)
                self._checkpoint_batches += 1
                if persisted:
                    self._checkpoint_entries += len(batch)
                else:
                    self._checkpoint_failures += 1
                self._in_flight_batch = None
            if self._closing:
                if self._pending:
                    self._checkpoint_event.set()
                    continue
                self._closed = True
                return
            if len(self._pending) >= self._max_pending_checkpoint_entries:
                self._checkpoint_event.set()

    async def _persist_batch(
        self,
        batch: Mapping[str, SerializedTrace],
    ) -> bool:
        try:
            entries = {
                key: CacheEntry(
                    schema=TRACE_RECORD_SCHEMA,
                    record=trace.model_dump(mode="json"),
                )
                for key, trace in batch.items()
            }
            await self._store.put_many(entries)
        except Exception:
            _LOGGER.warning(
                "trace cache checkpoint failed; dropping retry state",
                exc_info=True,
            )
            return False
        return True


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "TraceCacheStats",
    "WindowedTraceCache",
]
