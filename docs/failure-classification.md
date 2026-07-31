# Failure classification

`dr-code classify-failures` selects preprocessing parse failures and measured
candidate-test failures with deterministic DuckDB queries. Selection caps run
in SQL; summary totals always describe the full uncapped matching population.
Corpus task context is decoded from the viewer's reserved context struct and
normalized with the viewer's recursive JSON-safe value contract before prompt
rendering. Measured test evidence includes the canonical schema-v5 function,
case-count, error, timeout, and coverage facts; the contractually null
non-measured failure fields are not projected.

The details JSONL starts with a schema-versioned experiment header. Schema 4
records the primary or correction phase, one-based attempt, whether correction
succeeded, and a bounded safe primary-validation failure for every repeat. Its
full SHA-256 identity binds the authenticated run and dataset, immutable
preprocessing and evaluation coordinates, provider transport and timeout,
model, the resolved provider package and shebang runtime closure, a frozen
sanitized environment identity, an explicit typed lane transport/generation
policy, repeat and aggregation policy, taxonomy semantics, the exact primary
and correction prompt templates, and parse/test selection limits. The bounded
closure digest is reverified before resume and every provider call. Secret
environment values and provider transport details are delivered when needed
but never enter the identity or artifact. Injected lanes without a canonical
policy are rejected. Item records bind that experiment identity to exact
failure coordinates, the authenticated task-content identity, and rendered
prompt bytes. Fully rendered primary and correction prompts have deterministic
delivery caps after JSON escaping and error rendering.

The default filename is the complete experiment SHA-256. An explicit path must
either be absent or contain the exact same experiment header. `--force`
never bypasses that check. Canonical output basenames beginning with `.` and
ending in `.publication` or `.lock` are reserved for internal state and are
rejected before locking. Forced recomputation checkpoints to a private staged
artifact and atomically replaces the prior complete artifact only after every
selected item finishes. Ordinary runs use the same staged checkpoint and resume
it after interruption, leaving the last published artifact untouched. A
per-output lock serializes collision checking, classification, recovery,
publication, hashing, and rollup publication. The CLI acquires this lock before
opening DuckDB. Collision validation includes the database owner-lock path.
Viewer database ownership independently spans the entire database lifetime, so
invocations targeting different outputs cannot write the same DuckDB file
concurrently.

The classifier first requires the analytics-owned registered descriptor to
match every supplied immutable content coordinate. It then completes its one
analytics scan into a bounded temporary SQLite spool. Exact selected inputs and
`(dataset_id, task_id, task_identity)` targets are frozen, deduplicated, and
authenticated before the provider is called; provider work, checkpoints,
publication, and rollups all consume only that spool. A preprocessing-only run
may still produce output-only classifications when its selected items have no
task identity, but it cannot turn unauthenticated corpus task strings into task
annotations. Registering or classifying one run replaces only that run's task
membership and leaves other registered runs active. Task rollups are replaced
in one DuckDB transaction over the exact selected task identities. Successful
aggregates upsert machine rows without overwriting human annotations.
All-failed aggregates remove a stale machine row only when its provenance names
the failure-classifier producer and the same experiment identity; tag links are
removed only after that owned row is deleted.

Publication uses a durable DuckDB intent between the staged-file fsync and the
public replace. While an intent is pending, task-annotation reads and exports
suppress machine rows from its producer and experiment; human rows remain
visible. The final rollup transaction requires and deletes that exact intent.
On reopen under the same output lock, prior output plus intended stage aborts
the unpublished attempt, while intended output without a stage finishes its
rollups. Missing or third-hash evidence is rejected and remains suppressed for
manual investigation. Intents are keyed by canonical output path, so one
experiment may intentionally publish to multiple explicit paths; visibility
remains suppressed until every pending path for that producer and experiment
is resolved.

## Validation checklist

- [ ] Run a small production descriptor through the configured subscription
  provider and inspect one parse and one candidate-test prompt/result.
- [ ] Confirm the emitted experiment identity exactly matches the full SHA-256
  suffix of the default details filename.
- [ ] Interrupt a forced production rerun and confirm the prior complete
  details bytes and their machine-rollup provenance remain usable.
- [ ] Review a capped rerun containing failed transport calls and confirm only
  its selected, experiment-owned machine task rows are removed.
- [ ] Exercise two concurrent invocations targeting one explicit path and
  confirm every published rollup records the final details SHA-256.
