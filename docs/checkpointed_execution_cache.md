# Checkpointed execution cache

The execution-result cache keeps its hot path entirely in memory and uses
`dr-store` only for bulk prefetch and checkpoint operations.

## Storage dependency

`dr-code` expects the persistent record cache to provide these batch
operations in addition to its point API:

```python
async def get_many(
    keys: Iterable[str], *, schema: str
) -> dict[str, CacheHit | None]: ...

async def put_many(
    entries: Mapping[str, CacheEntry],
) -> dict[str, ObjectReference]: ...
```

`get_many` returns every distinct requested key, using `None` for a miss and a
verified `CacheHit` otherwise. Each `CacheEntry` passed to `put_many` carries
its own `schema` and `record`. The write validates and prepares the complete
batch before mutation, commits its object rows and bindings in one transaction,
and returns the stored first-writer winner for each key.

## Execution-cache behavior

- Execution keys use compact, precomputed digests for candidate source and
  test input together with computation, timeout, harness, and runtime identity.
- The metrics engine supplies all planned keys to the cache before point
  lookups. A persistent implementation bulk-loads only previously unseen keys;
  all subsequent `get` and `put` operations use memory.
- `score_humaneval_submissions_batch` extracts and plans every submission
  before one awaited `run_requests` call, so one cache prefetch covers the
  complete scoring batch. The single-submission scorer delegates to that batch
  boundary.
- Newly computed outcomes remain dirty until a checkpoint snapshots them for
  one background writer task. Only one checkpoint is in flight, and later
  writes are coalesced into the next batch.
- Checkpoints are triggered by explicit task boundaries or a configured number
  of new outcomes, not elapsed time. Normal close drains a final checkpoint.
- Persistent-cache read and write failures are observable but do not fail an
  evaluation. A failed write remains eligible for a later checkpoint while the
  cache remains open. Process termination may lose dirty or in-flight entries.
- Runtime identity is mandatory for persistent reuse. The injected executor
  itself is not serialized into the key.
- The caller restricts persistent reuse to workloads whose outcomes it treats
  as stable within that runtime scope and coordinates one active writer for the
  scope. Concurrent writers are unsupported because conflicting winners are
  not reconciled.

The checkpoint scheduler, execution schemas, identity inputs, and outcome
validation belong to `dr-code`; `dr-store` remains domain-agnostic.
