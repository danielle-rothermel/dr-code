# Windowed trace cache

The preprocessing trace cache keeps one explicitly bounded window in memory
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

## Trace-cache behavior

- Trace keys bind raw input text, the resolved preprocessing producer
  coordinate, and the trace schema version through
  `preprocessing_trace_cache_key`.
- A caller prefetches at most one declared cache window. Hits and explicit
  misses occupy bounded resident entries until the caller discards them.
- Cached traces are validated on restore; wrong producer or input degrades to a
  miss and reruns through the pool.
- Newly computed traces enter at most one bounded pending checkpoint batch. At
  most one other batch is in flight. Entries beyond the pending bound remain
  memory-only.
- Checkpoints are triggered by state, not elapsed time. Normal close drains the
  pending batch.
- Persistent read and write failures are logged and degrade to misses or dropped
  retry state rather than failing the batch.

## Parallel execution

`preprocess_batch` combines `WindowedTraceCache` with dr-exec 0.1.8's
`ImportableJsonExecutor` and `ExecutionPool`. Worker count bounds concurrent
in-process preprocessing jobs the same way candidate execution bounds process
jobs, without subprocess overhead.

The cache schemas, identity inputs, bounds, and trace validation belong to
`dr-code`; `dr-store` and `dr-exec` remain domain-agnostic.
