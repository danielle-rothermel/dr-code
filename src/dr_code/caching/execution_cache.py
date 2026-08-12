from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, Protocol, runtime_checkable

from dr_serialize import canonical_json_bytes, identity_document_hash
from dr_store import (
    CacheEntry,
    CacheHit,
    EvictStatus,
    ObjectReference,
    derive_cache_key,
)

from dr_code.core.models import FrozenModel
from dr_code.evaluation.identity import EvaluationRuntimeIdentity
from dr_code.evaluation.records import CandidateExecutionOutcome
from dr_code.evaluation.references import StoredRecordReference

CACHED_EXECUTION_OBSERVATION_SCHEMA_VERSION: Final = 1
EXECUTION_CACHE_NAMESPACE: Final = "dr-code/evaluation-execution-v1"
EXECUTION_CACHE_RECORD_SCHEMA: Final = (
    "dr-code/cached-execution-observation-v1"
)
_LOGGER = logging.getLogger(__name__)


class BatchRecordStore(Protocol):
    """The bounded batch record-store operations required by the cache."""

    async def get_many(
        self,
        keys: Iterable[str],
        *,
        schema: str,
    ) -> dict[str, CacheHit | None]: ...

    async def put_many(
        self,
        entries: Mapping[str, CacheEntry],
    ) -> dict[str, ObjectReference]: ...


@runtime_checkable
class EvictableBatchRecordStore(BatchRecordStore, Protocol):
    """Batch record store that also supports cache-grade binding eviction."""

    async def evict_bindings(
        self,
        keys: Iterable[str],
    ) -> dict[str, EvictStatus]: ...


class CachedExecutionObservation(FrozenModel):
    schema_version: Literal[1] = CACHED_EXECUTION_OBSERVATION_SCHEMA_VERSION
    source_record: StoredRecordReference
    outcome: CandidateExecutionOutcome


@dataclass(frozen=True, slots=True)
class ExecutionCacheStats:
    prefetched_entries: int
    memory_hits: int
    memory_misses: int
    dirty_entries: int
    checkpoint_batches: int
    checkpoint_entries: int
    checkpoint_failures: int
    in_flight: bool


class WindowedExecutionCache:
    """Bounded resident observations with bounded best-effort checkpointing."""

    def __init__(
        self,
        store: BatchRecordStore,
        *,
        runtime: EvaluationRuntimeIdentity,
        max_resident_entries: int,
        max_pending_checkpoint_entries: int,
    ) -> None:
        _require_positive_int(max_resident_entries, "max_resident_entries")
        _require_positive_int(
            max_pending_checkpoint_entries,
            "max_pending_checkpoint_entries",
        )
        self._store = store
        self._runtime_digest = str(identity_document_hash(runtime.document))
        self._max_resident_entries = max_resident_entries
        self._max_pending_checkpoint_entries = max_pending_checkpoint_entries
        self._resident: dict[str, CachedExecutionObservation | None] = {}
        self._pending: dict[str, CachedExecutionObservation] = {}
        self._in_flight_batch: dict[str, CachedExecutionObservation] | None = (
            None
        )
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
            name="dr-code-windowed-execution-cache-writer",
        )

    async def __aenter__(self) -> WindowedExecutionCache:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def prefetch(self, request_keys: Iterable[str], /) -> None:
        self._require_open()
        keys = tuple(dict.fromkeys(request_keys))
        unseen = tuple(key for key in keys if key not in self._resident)
        if len(self._resident) + len(unseen) > self._max_resident_entries:
            raise ValueError(
                "execution cache prefetch exceeds max_resident_entries"
            )
        if not unseen:
            return

        persistent_to_request = {
            self._persistent_key(key): key for key in unseen
        }
        try:
            hits = await self._store.get_many(
                tuple(persistent_to_request),
                schema=EXECUTION_CACHE_RECORD_SCHEMA,
            )
        except Exception:
            _LOGGER.warning(
                "execution cache prefetch failed; treating entries as misses",
                exc_info=True,
            )
            hits = {}

        restored: dict[str, CachedExecutionObservation | None] = {}
        for persistent_key, request_key in persistent_to_request.items():
            hit = hits.get(persistent_key)
            if hit is None:
                restored[request_key] = None
                continue
            try:
                restored[request_key] = (
                    CachedExecutionObservation.model_validate_json(
                        canonical_json_bytes(hit.record),
                        strict=True,
                    )
                )
            except Exception:
                _LOGGER.warning(
                    "invalid execution cache entry; treating it as a miss",
                    exc_info=True,
                )
                restored[request_key] = None

        self._require_open()
        for key, observation in restored.items():
            if key in self._resident:
                continue
            self._resident[key] = observation
            if observation is not None:
                self._prefetched_entries += 1

    def get(self, request_key: str, /) -> CachedExecutionObservation | None:
        self._require_open()
        observation = self._resident.get(request_key)
        if observation is None:
            self._memory_misses += 1
        else:
            self._memory_hits += 1
        return observation

    async def put(
        self,
        request_key: str,
        observation: CachedExecutionObservation,
        /,
    ) -> None:
        self._require_open()
        if (
            request_key not in self._resident
            and len(self._resident) >= self._max_resident_entries
        ):
            raise ValueError(
                "execution cache put exceeds max_resident_entries"
            )
        self._resident[request_key] = observation

        if request_key in self._pending:
            self._pending[request_key] = observation
        elif len(self._pending) < self._max_pending_checkpoint_entries:
            self._pending[request_key] = observation
        if (
            len(self._pending) >= self._max_pending_checkpoint_entries
            and self._in_flight_batch is None
        ):
            self._checkpoint_event.set()

    def discard(self, request_key: str, /) -> None:
        self._require_open()
        self._resident.pop(request_key, None)

    async def evict(
        self,
        request_keys: Iterable[str],
        /,
    ) -> dict[str, EvictStatus]:
        """Remove persisted bindings and in-memory state for request keys."""

        self._require_open()
        keys = tuple(dict.fromkeys(request_keys))
        if not keys:
            return {}

        persistent_to_request = {
            self._persistent_key(key): key for key in keys
        }
        for request_key in keys:
            self._resident.pop(request_key, None)
            self._pending.pop(request_key, None)
            if self._in_flight_batch is not None:
                self._in_flight_batch.pop(request_key, None)

        if not isinstance(self._store, EvictableBatchRecordStore):
            raise TypeError(
                "execution cache eviction requires a store that implements "
                "evict_bindings"
            )
        statuses = await self._store.evict_bindings(
            persistent_to_request.keys()
        )
        return {
            persistent_to_request[persistent_key]: status
            for persistent_key, status in statuses.items()
        }

    def stats(self) -> ExecutionCacheStats:
        return ExecutionCacheStats(
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

    def _persistent_key(self, request_key: str) -> str:
        return derive_cache_key(
            EXECUTION_CACHE_NAMESPACE,
            {
                "request_key": request_key,
                "runtime_identity": self._runtime_digest,
            },
        )

    def _require_open(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("execution cache is closed")

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
        batch: Mapping[str, CachedExecutionObservation],
    ) -> bool:
        try:
            entries = {
                self._persistent_key(key): CacheEntry(
                    schema=EXECUTION_CACHE_RECORD_SCHEMA,
                    record=observation.model_dump(mode="json"),
                )
                for key, observation in batch.items()
            }
            await self._store.put_many(entries)
        except Exception:
            _LOGGER.warning(
                "execution cache checkpoint failed; dropping retry state",
                exc_info=True,
            )
            return False
        return True


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "BatchRecordStore",
    "CACHED_EXECUTION_OBSERVATION_SCHEMA_VERSION",
    "CachedExecutionObservation",
    "EvictableBatchRecordStore",
    "EXECUTION_CACHE_NAMESPACE",
    "EXECUTION_CACHE_RECORD_SCHEMA",
    "ExecutionCacheStats",
    "WindowedExecutionCache",
]
