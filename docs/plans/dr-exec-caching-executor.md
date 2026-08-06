# CachingExecutor for dr-exec

Status: plan only. Nothing here is implemented, and it is not implementable
until dr-code's execution path runs through dr-exec.

## Goal

Durable, fleet-wide memoization of execution outcomes. Re-running the same
declared payload against the same declared runtime should read a stored result
rather than spawn a child. Evaluation sweeps repeat identical HumanEval batch
requests across reruns, machines, and developers; today every repeat pays full
process cost for a result that is already known.

## Placement

`dr_exec/capabilities/caching.py`, a sibling of `capabilities/fake.py`.

`CachingExecutor` implements the `Executor` protocol by wrapping an inner
`Executor`. It spawns nothing itself, decides no outcome itself, and adds no
method to the boundary: a caller holds an `Executor` and cannot tell from the
type whether caching is in play. Wrapping — rather than an option on
`ProcessExecutor` — keeps the production executor's containment and recording
claims untouched, and lets the conformance suite run the wrapper as an
executor in its own right.

## Key

One entry is identified by the pair:

- the canonical declaration digest, `recording/identity.py`
  `_canonical_declaration_digest`, which already exists as the stable identity
  of what was declared; and
- the hash of the isolated-host runtime identity document, so a result is
  never replayed across a different declared runtime.

Both are derived with dr-serialize `json_hash` over the canonical JSON
profile, so the key has exactly one canonicalization. The composed key is
stored through the dr-store cache primitive (`derive_cache_key` plus
`RecordCache`), which is the same best-effort read/first-writer-wins facade
dr-code's trace cache uses.

The key payload carries a version segment. Changing what the key covers, or
what the value means, is a version bump: old entries stop being addressable
instead of being reinterpreted. There is no invalidation path other than
re-keying — no TTL, no eviction, no delete.

## Value

The JSON projection of the completed execution's result record — the same
projection the durable run record already persists, so nothing new has to be
made serializable.

A hit is replayed as a `CompletedExecution` carrying an explicit cached
receipt: a distinct receipt kind alongside `CompleteRecordReceipt`,
`DegradedRecordReceipt`, and `FakeRecordReceipt`. A replayed completion must
never be mistakable for a run that actually happened, and must never claim a
record directory this process did not write. The receipt names the entry it
came from.

## Policy

- Never cache channel-attributed or executor-attributed failures. dr-exec
  contractually treats evidence-attributed channel failures as the retriable
  class, and executor attribution is the default for unknown causes; caching
  either would make a transient or unexplained fault permanent.
- Cache exited outcomes. An exit status for a declared payload under a declared
  runtime is the outcome the cache exists to serve, and exit interpretation is
  caller policy applied after the fact.
- Budget-exceeded outcomes are cacheable only behind an explicit flag, default
  off. They are load-dependent: the same payload can exceed a wall-time budget
  on a busy machine and finish on an idle one, so replaying one asserts
  something about the host, not about the payload.
- Signalled, spawn-absent, spawn-failed, protocol-failed, and cancelled
  outcomes are out of scope for a first cut; they are not what the cache is
  for and each needs its own argument.
- A cache read is best-effort. Any storage-level fault reads as a miss and
  falls through to the inner executor, so a broken cache degrades cost, not
  correctness.

## Honesty note

A hit certifies **same declared runtime**: the resolved interpreter path,
version, and platform recorded in the isolated-host runtime identity document.
It does not certify that the interpreter, standard library, or installed
package bytes are the same as when the entry was written. A host that swaps
what lives at a resolved interpreter path produces a stale hit, and this design
cannot detect that.

The upgrade path is `docs/future-plans/verified-python-runtime.md` in dr-exec.
A verified runtime's identity covers the lockfile, provisioning inputs,
interpreter and standard-library bytes, installed package bytes, and invocation
mode. That identity slots into the runtime half of the key with no redesign:
the key shape is already "declaration digest x runtime identity hash", and a
stronger runtime identity is a stronger second factor plus a key version bump.
Until then, the cached-receipt kind and the documented claim are what keep the
weaker guarantee visible at the call site.

## Validation

- Run the shared executor conformance suite against a `CachingExecutor`
  wrapping a `FakeExecutor`. The wrapper is an `Executor`, so it owes the
  boundary's promises — declaration parity, thread safety, outcome shape, and
  receipt kind — and the suite is what qualifies it.
- Targeted tests beyond conformance: a miss delegates exactly once and stores;
  a hit delegates zero times and returns the cached receipt; a changed
  declaration and a changed runtime identity each miss; each non-cacheable
  attribution class stores nothing and delegates on every call; budget-exceeded
  is skipped by default and cached with the flag; a corrupt or unreadable entry
  reads as a miss.
- Concurrency: two threads racing the same key both get a correct completion,
  with first-writer-wins leaving one entry.

## Sequencing

1. dr-code's execution path cuts over to dr-exec. dr-code PR #91 records the
   adoption requirements; until that lands there is no dr-exec caller whose
   repeats this would serve.
2. Then this executor, as an additive capability with no change to
   `ProcessExecutor`, `FakeExecutor`, or the `Executor` protocol.
