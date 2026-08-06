from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from threading import Condition, Thread
from typing import Final, Protocol

from dr_serialize import IdentityDocument, identity_document_hash
from dr_store import CacheEntry, CacheHit, ObjectReference, derive_cache_key

from dr_code.metrics.engine.execution import ExecutionOutcome

_EXECUTION_CACHE_NAMESPACE: Final = "dr-code/execution-cache@1"
_EXECUTION_OUTCOME_SCHEMA: Final = "dr-code/execution-outcome@1"
_LOGGER = logging.getLogger(__name__)


class BatchRecordStore(Protocol):
    """The batch record-store operations required by execution caching."""

    def get_many(
        self,
        keys: Iterable[str],
        *,
        schema: str,
    ) -> dict[str, CacheHit | None]: ...

    def put_many(
        self,
        entries: Mapping[str, CacheEntry],
    ) -> dict[str, ObjectReference]: ...


@dataclass(frozen=True, slots=True)
class ExecutionCacheStats:
    """An immutable point-in-time view of execution-cache activity."""

    prefetched_entries: int
    memory_hits: int
    memory_misses: int
    dirty_entries: int
    checkpoint_batches: int
    checkpoint_entries: int
    checkpoint_failures: int
    in_flight: bool


class CheckpointedExecutionCache:
    """Bulk-persistent execution outcomes with a memory-only hot path.

    The runtime identity is caller-owned because it covers behavior outside
    the request itself, including the Python runtime, harness, and dependency
    environment. The injected executor is deliberately not persisted. Callers
    restrict reuse to stable workloads and coordinate one writer per scope;
    conflicting first-writer winners are not reconciled.
    """

    def __init__(
        self,
        store: BatchRecordStore,
        *,
        runtime_identity: IdentityDocument,
        checkpoint_entry_count: int = 1_000,
    ) -> None:
        if not isinstance(runtime_identity, IdentityDocument):
            raise TypeError("runtime_identity must be an IdentityDocument")
        if (
            isinstance(checkpoint_entry_count, bool)
            or not isinstance(checkpoint_entry_count, int)
            or checkpoint_entry_count <= 0
        ):
            raise ValueError(
                "checkpoint_entry_count must be a positive integer"
            )

        self._store = store
        self._runtime_digest = str(identity_document_hash(runtime_identity))
        self._checkpoint_entry_count = checkpoint_entry_count
        self._condition = Condition()
        self._outcomes: dict[str, ExecutionOutcome] = {}
        self._seen_keys: set[str] = set()
        self._dirty: dict[str, ExecutionOutcome] = {}
        self._checkpoint_requested = False
        self._closing = False
        self._closed = False
        self._in_flight = False
        self._prefetched_entries = 0
        self._memory_hits = 0
        self._memory_misses = 0
        self._checkpoint_batches = 0
        self._checkpoint_entries = 0
        self._checkpoint_failures = 0
        self._writer = Thread(
            target=self._write_checkpoints,
            name="dr-code-execution-cache-writer",
            daemon=True,
        )
        self._writer.start()

    def __enter__(self) -> CheckpointedExecutionCache:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def prefetch(self, keys: Sequence[str]) -> None:
        """Bulk-load previously unseen keys without affecting evaluation."""
        with self._condition:
            self._require_open()
            unseen: list[str] = []
            for key in keys:
                if key in self._seen_keys:
                    continue
                self._seen_keys.add(key)
                unseen.append(key)

        if not unseen:
            return

        persistent_to_request = {
            self._persistent_key(key): key for key in unseen
        }
        try:
            hits = self._store.get_many(
                tuple(persistent_to_request),
                schema=_EXECUTION_OUTCOME_SCHEMA,
            )
        except Exception:
            _LOGGER.warning(
                "execution cache prefetch failed; treating entries as misses",
                exc_info=True,
            )
            return

        restored: dict[str, ExecutionOutcome] = {}
        for persistent_key, request_key in persistent_to_request.items():
            hit = hits.get(persistent_key)
            if hit is None:
                continue
            try:
                restored[request_key] = ExecutionOutcome.model_validate(
                    hit.record,
                    strict=True,
                )
            except Exception:
                _LOGGER.warning(
                    "invalid execution cache entry; treating it as a miss",
                    exc_info=True,
                )

        with self._condition:
            for key, outcome in restored.items():
                # A computation that completed while prefetch was in progress
                # is newer local evidence and must not be overwritten.
                if key in self._outcomes:
                    continue
                self._outcomes[key] = outcome
                self._prefetched_entries += 1

    def get(self, key: str) -> ExecutionOutcome | None:
        """Return a memory hit or miss without consulting persistence."""
        with self._condition:
            self._require_open()
            self._seen_keys.add(key)
            outcome = self._outcomes.get(key)
            if outcome is None:
                self._memory_misses += 1
            else:
                self._memory_hits += 1
            return outcome

    def put(self, key: str, outcome: ExecutionOutcome) -> None:
        """Update memory and schedule an entry-count checkpoint if needed."""
        with self._condition:
            self._require_open()
            self._seen_keys.add(key)
            self._outcomes[key] = outcome
            self._dirty[key] = outcome
            if len(self._dirty) >= self._checkpoint_entry_count:
                self._request_checkpoint()

    def checkpoint(self) -> None:
        """Schedule all currently dirty entries for background persistence."""
        with self._condition:
            self._require_open()
            self._request_checkpoint()

    def close(self) -> None:
        """Drain one final checkpoint and stop the writer thread."""
        with self._condition:
            if self._closed:
                return
            self._closing = True
            self._request_checkpoint()
        self._writer.join()

    def stats(self) -> ExecutionCacheStats:
        with self._condition:
            return ExecutionCacheStats(
                prefetched_entries=self._prefetched_entries,
                memory_hits=self._memory_hits,
                memory_misses=self._memory_misses,
                dirty_entries=len(self._dirty),
                checkpoint_batches=self._checkpoint_batches,
                checkpoint_entries=self._checkpoint_entries,
                checkpoint_failures=self._checkpoint_failures,
                in_flight=self._in_flight,
            )

    def _persistent_key(self, request_key: str) -> str:
        return derive_cache_key(
            _EXECUTION_CACHE_NAMESPACE,
            {
                "request_key": request_key,
                "runtime_identity": self._runtime_digest,
            },
        )

    def _request_checkpoint(self) -> None:
        self._checkpoint_requested = True
        self._condition.notify()

    def _require_open(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("execution cache is closed")

    def _write_checkpoints(self) -> None:
        while True:
            with self._condition:
                while not self._checkpoint_requested:
                    if self._closing:
                        self._closed = True
                        self._condition.notify_all()
                        return
                    self._condition.wait()

                self._checkpoint_requested = False
                if not self._dirty:
                    if self._closing:
                        self._closed = True
                        self._condition.notify_all()
                        return
                    continue
                batch = self._dirty
                self._dirty = {}
                self._in_flight = True

            persisted = self._persist_batch(batch)

            with self._condition:
                self._checkpoint_batches += 1
                self._in_flight = False
                if persisted:
                    self._checkpoint_entries += len(batch)
                else:
                    self._checkpoint_failures += 1
                    # Outcomes computed during the failed write win over the
                    # older failed snapshot for the same request key.
                    self._dirty = batch | self._dirty
                self._condition.notify_all()

    def _persist_batch(
        self,
        batch: Mapping[str, ExecutionOutcome],
    ) -> bool:
        try:
            entries = {
                self._persistent_key(key): CacheEntry(
                    schema=_EXECUTION_OUTCOME_SCHEMA,
                    record=outcome.model_dump(mode="json"),
                )
                for key, outcome in batch.items()
            }
            self._store.put_many(entries)
        except Exception:
            _LOGGER.warning(
                "execution cache checkpoint failed; retaining entries",
                exc_info=True,
            )
            return False
        return True


__all__ = [
    "BatchRecordStore",
    "CheckpointedExecutionCache",
    "ExecutionCacheStats",
]
