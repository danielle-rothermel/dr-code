# Windowed execution cache

The candidate-execution cache keeps one explicitly bounded window in memory
and uses `dr-store` only for bounded bulk prefetch and best-effort checkpoint
operations.

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
its own `schema` and `record`.

## Execution-cache behavior

- Candidate execution keys bind the complete evaluator request, job budget,
  caller cache namespace, and runtime identity.
- A caller prefetches at most one declared cache window. Hits and explicit
  misses occupy bounded resident entries until the caller discards them. An
  evaluation batch request that declares itself fresh skips both prefetch and
  lookup for its generations, so every candidate re-executes. Because
  persisted bindings are first-writer-wins, a fresh outcome does not replace
  an entry already stored under the same key.
- Cached observations retain the interpreted candidate outcome and a portable
  reference to the source executed record. A cache hit produces reused
  provenance; it never claims that a new process ran.
- Only candidate-owned outcomes are written: a completed job and a
  candidate-owned termination. Harness and infrastructure failures describe
  the run rather than the candidate and never enter the cache.
- Newly computed observations eligible for persistence enter at most one
  bounded pending checkpoint batch. At most one other batch is in flight.
  Entries beyond the pending bound remain memory-only.
- Checkpoints are triggered by state, not elapsed time. Normal close drains the
  pending batch.
- Persistent read and write failures are observable but do not fail an
  evaluation. Failed writes are dropped rather than accumulated for unbounded
  retry.
- Runtime identity is mandatory for persistent reuse. The injected executor
  itself is not serialized into the key.

The cache schemas, identity inputs, bounds, and observation validation belong
to `dr-code`; `dr-store` remains domain-agnostic.
