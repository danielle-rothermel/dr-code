# Future improvements

## Evaluation execution and storage throughput

The evaluation core keeps one materialized candidate in one process-bound
`dr-exec` job and uses the released directory run store and singular
content-addressed object operations. These boundaries preserve candidate-level
attribution but do not establish suitability for sustained high-volume
workloads.

Potential dependency improvements are:

- A persistent or reusable-worker `dr-exec` implementation that preserves one
  logical job and result per candidate while amortizing interpreter startup and
  evaluator imports. Its design must define state reset, poisoned-worker
  replacement, cancellation, cleanup, and isolation before reuse is safe.
- A high-volume `dr-exec` run store with an extensible portable reference
  variant and explicit retention ownership. It must avoid one directory and
  multiple filesystem artifacts per execution while preserving the run-record
  lifecycle and audit boundary.
- A public bounded `dr-store` artifact-bundle operation that returns the
  validated manifest payload while consuming one selected verified artifact.
  Selective evaluation readers could then compare an artifact's source binding
  with the bundle payload without auditing or restoring unrelated artifacts.
- Caller-bounded `dr-store.ObjectStore` reads that enforce byte and JSON-depth
  limits before materializing and decoding stored canonical text. Evaluation v1
  can bound reference counts, sequential reads, and post-decode schemas, but it
  cannot claim preallocation bounds from the released singular `get` API.
- Bounded set-based `dr-store.ObjectStore` reads and writes for authoritative
  content-addressed records. Operations should deduplicate references, use
  backend-sized chunks, and avoid both N+1 round trips and unbounded database
  fan-out.
- Sustained-throughput qualification for process execution, run recording,
  content-addressed record access, and artifact publication. Qualification
  should report throughput, resident memory, storage growth, and cleanup under
  realistic request and output sizes without creating a numeric service-level
  guarantee.

These improvements become relevant when measured process startup, directory
recording, or singular object-store operations materially limit an application.
They do not justify batching multiple candidates into one process or weakening
candidate-level outcome attribution in the evaluation core.

## Projection comparison evidence

Comparing aggregation or score projections requires an explicit resolver for
the projection evidence rows. Version one returns `ProjectionNotComparable`
when the rows are unavailable instead of loading them speculatively.
