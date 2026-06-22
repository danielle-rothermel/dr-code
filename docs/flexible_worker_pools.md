# Flexible Worker Pools

Status date: 2026-06-22.

This document tracks whether `dr-code` can build flexible parse/test worker
pools on top of the current sibling `../dr-queues` checkout, and what still
needs to change before that workflow is safe for larger pool runs.

## Current Answer

`dr-queues` is now ready to build on for `dr-code`'s flexible worker-pool
workflow, as a local editable dependency. The core runtime blockers from the
original plan have moved out of `dr-queues`: run manifests, seed batches,
worker records, job states, status, wait, queue snapshots, stage lifecycle, and
attach-to-existing-run validation now live in Mongo-backed runtime APIs.

The remaining work is mostly `dr-code` integration hardening:

- Treat MongoDB as the source of truth for all continuation state.
- Use `dr-queues` status/lifecycle/replay APIs consistently instead of direct
  Mongo queries in operational code.
- Tighten run continuation checks around seed source, task set, worker plan,
  exports, and duplicate seeding.
- Add a focused end-to-end smoke pass for the workers-before-seed split
  lifecycle workflow before scaling beyond proof-sized runs.

## Current `dr-queues` State

The sibling `../dr-queues` repo has advanced beyond the assumptions in the
original version of this doc.

Implemented and relevant now:

- Mongo-backed run manifests, seed batches, worker records, latest job states,
  failure attempts, target holds, and append-only pipeline events.
- No filesystem-backed runtime store for new runs. New run continuation state
  should not depend on `.runs/{run_id}`.
- `setup_run_queues`, `attach_run_queues`, `seed_run`, and `run_in_process`
  for setup, attach, seed, and in-process execution.
- `dr-queues-run init`, `seed`, `status`, `wait`, `start`, `replace`, `stop`,
  `workers`, `failures`, `attempts`, `holds`, and `replay`.
- `get_run_status` aggregates Mongo progress, latest job states, active worker
  records, and RabbitMQ queue snapshots.
- `wait_for_run` is resume-aware in the important sense for `dr-code`: it
  initializes from persisted Mongo state and can wait for a named stage or for
  terminal completion.
- Terminal waiting for detached runs starts a `TerminalTap`, so final-stage
  completed messages are converted into terminal events without an in-process
  driver.
- Detached worker lifecycle is stage-specific and supports start, replace,
  stop, list, heartbeats, stale-worker detection, include/exclude selectors,
  and multiple worker records per stage.
- Seed work is protected by Mongo seed batches and duplicate job-id detection.
- Stage eligibility and manual replay exist for pending, held, retry-waiting,
  failed, and dead-lettered work.
- Runtime observability and a local viewer exist for summaries, queue depths,
  worker records, holds, blocked jobs, attempts, and recent events.

Important constraints still present:

- RabbitMQ remains the durable message transport; Mongo is the state/query
  layer. Both services must be available for real runs.
- Handler registration is still downstream-project specific. `dr-code` must
  pass `dr_code.pipeline.handlers` for detached workers.
- Replay and hold expiry are manual. There is no background retry scheduler or
  token-bucket provider throttling yet.
- Status/wait completeness is based on expected jobs from active seed batches.
  A run with no seed batch has expected count zero and can appear complete.
- Worker lifecycle is local-process oriented: `stop` signals workers only on
  the current host, while records for other hosts are stop-requested in Mongo.

## Current `dr-code` State

`dr-code` already depends on `dr-queues` through `[tool.uv.sources]` as an
editable sibling path:

```toml
dr-queues = { path = "../dr-queues", editable = true }
```

The newer `dr-code` branch has also implemented most of the command split that
the original plan called for:

- `scripts/eval_run.py init` creates a Mongo-backed run manifest and queues.
- `scripts/eval_run.py seed` attaches to an existing manifest, loads attempts,
  builds `JobEnvelope`s, and publishes parse-stage work.
- `scripts/eval_run.py start` starts detached parse and/or test workers through
  `dr_queues.start_stage_workers`.
- `scripts/eval_run.py stop` requests worker stops by stage or worker id.
- `scripts/eval_run.py wait` waits for terminal or named-stage completion via
  `dr_queues.wait_for_run`.
- `scripts/eval_run.py status` reports persisted run status.
- `scripts/eval_run.py export` reconstructs attempts, parse JSONL, test JSONL,
  manifest JSON, and final proof report from persisted state.
- `scripts/eval_run.py run` still provides the one-shot preflight, seed,
  execute, wait, export, and report path for proof runs.
- `docs/adr/0001-mongo-backed-evaluation-run-state.md` records that MongoDB is
  the lifecycle source of truth and exported files are derived artifacts.
- The pipeline runbook and overview now describe Mongo-backed lifecycle state,
  split lifecycle commands, and the 2026-06-22 manual smoke verification.

Gaps that remain in `dr-code`:

- `pipeline.tune` still counts terminals, stage completions, and infra errors
  through direct Mongo queries. It should move to `get_run_status` and
  `MongoRunStore` where possible, leaving only `eval_results` queries in
  `dr-code`.
- The split commands do not yet persist or validate eval-specific seed metadata
  such as dump directory, task indices, limit, attempts path, and source hash.
  `dr-queues` validates the pipeline definition and job IDs, but not those
  `dr-code` domain constraints.
- `start_eval_workers` selects stages but does not yet expose target selectors,
  replay, holds, failures, or attempts through the `dr-code` CLI.
- `status` output is intentionally minimal and does not yet include the full
  eval-oriented view: seeded attempts, parse outputs, test terminals,
  failed/held/retry jobs, active worker concurrency, stale workers, and queue
  depths in one summary.
- `export` supports partial parse/test JSONL, but proof report generation is
  final-only. There is not yet a named partial report for in-flight runs.
- The one-shot detached path starts workers after seeding. The split commands
  can support "workers first, seed later", but that recipe still needs a smoke
  test.

## Feature Status

| Feature | Current status | Notes |
|---------|----------------|-------|
| Stage-selective execution | Mostly supported | `eval_run.py start --stage parse` and repeated `--stage` work; one-shot `run` remains full-pipeline. |
| Reusable/resumable run IDs | Mostly supported | `attach_run_queues` validates the pipeline definition and Mongo state persists across commands. `dr-code` still needs eval-specific seed metadata validation. |
| Decoupled seeding from worker startup | Supported by runtime, needs workers-before-seed smoke in `dr-code` | Workers can idle on empty queues and seeding is separate. Seed-before-workers split lifecycle passed on 2026-06-22. |
| Blocking idle workers | Supported | RabbitMQ consumers wait on empty queues until stopped. |
| Independent stage scaling | Supported | Worker specs like `parse=8,test=2` are parsed in both repos. |
| Stage-specific lifecycle control | Supported at runtime, partially surfaced in `dr-code` | Start/stop by stage exists. Replace, worker listing, holds, replay, and failure views are not all wrapped by `dr-code`. |
| Incremental continuation | Mostly supported | Init, seed, start, wait, status, and export are split. Stronger `dr-code` metadata checks are still needed. |
| Partial completion awareness | Supported by runtime, partially surfaced | `wait --target parse` and per-stage completion counts exist. `dr-code` should improve status/export UX. |
| Pipeline-shaped queue topology | Supported | Parse and test remain separate pipeline stages, with stage output queues chained into downstream input. |
| Dynamic worker scaling | Supported | `dr-queues` has start and replace. `dr-code` tuning uses replace for test workers; additive scale-up can use start but should be documented. |
| Failure hold/retry/replay controls | Runtime supported, not eval-wrapped | Useful for provider-like target partitions later; immediate value is replaying failed/held jobs manually. |
| Observability viewer | Runtime supported | `dr-queues-viewer` can inspect `dr-code` runs because they use the shared Mongo runtime state. |

## Plan Impact

The original high-level plan does not need to change: `dr-code` should still
build flexible parse/test eval orchestration on top of `dr-queues`, and should
still keep parse/test domain semantics in `dr-code`.

The implementation plan should change in three ways:

1. Do less in `dr-queues`. The generic runtime work that this doc used to ask
   for is mostly done.
2. Move effort to `dr-code` hardening: continuation metadata, status UX,
   runbook updates, split-command smoke tests, and cleanup of stale direct
   Mongo/runtime assumptions.
3. Treat exports as derived snapshots. Do not rebuild any `.runs`-style
   filesystem state layer in `dr-code`.

## Safe Local Dependency Adoption Checklist

Use the latest `../dr-queues` safely in `dr-code` by completing these updates:

1. Keep `dr-queues` as the editable local dependency for now, and run `uv sync`
   in `dr-code` after changes in the sibling repo.
2. Keep runbook and overview references aligned with Mongo `run_manifests`,
   not `.runs/{run_id}` filesystem manifests.
3. Prefer `uv run scripts/eval_run.py start/stop` recipes in `dr-code` docs;
   use lower-level `dr-queues-run replace` only for controls not wrapped by
   `dr-code`.
4. Add a workers-before-seed smoke check with a tiny attempts fixture.
5. Add a continuation smoke check:
   initialize once, seed once, stop/restart workers, wait, export, and confirm
   terminal count equals `MongoRunStore.expected_job_count(run_id)`.
6. Persist eval-specific run metadata during init or seed: seed source,
   attempts path or dump dir, task indices, limit, expected task/sample counts,
   and enough source identity to detect accidental continuation against the
   wrong input.
7. Refuse duplicate or incompatible `seed` operations in `dr-code` before
   calling `seed_run`; rely on `dr-queues` duplicate job-id protection as the
   lower-level guard, not the user-facing explanation.
8. Update `pipeline.tune` to use `get_run_status` / `MongoRunStore` for
   terminal and stage progress, keeping only `eval_results` outcome counts in
   the `dr-code` Mongo adapter.
9. Surface `dr-queues` worker records in `scripts/eval_run.py status`,
    including active/stale/stop-requested workers and active concurrency by
    stage.
10. Add `scripts/eval_run.py workers` or document `dr-queues-run workers` as
    the supported worker-list command.
11. Decide whether `dr-code` should wrap `dr-queues` `replace`, `replay`,
    `failures`, `attempts`, and `holds`, or explicitly tell operators to use
    `dr-queues-run` for those lower-level controls.
12. Verify partial export behavior on a parse-complete/test-incomplete run and
    document that `proof_report.json` appears only after terminal completion.
13. Ensure detached worker commands always pass
    `--handlers-module dr_code.pipeline.handlers`.
14. Smoke `dr-queues-viewer --run-id <run_id>` against a `dr-code` run and add
    it to the runbook if useful.

## Near-Term Recommended Sequence

1. Run the split lifecycle on a tiny fixture with workers started before seed.
2. Add eval-specific metadata validation before trusting continuation for large
   pool runs.
3. Move tuning progress reads onto `dr-queues` runtime status.
4. Run a medium detached proof with the split commands rather than the one-shot
   driver, then freeze the operational recipe for full-pool replay.
