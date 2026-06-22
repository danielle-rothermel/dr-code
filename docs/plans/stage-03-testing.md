# Stage 3 — Testing

[← Overview](./overview.md) · **Status:** Done (2026-06-21). Adapter in `src/dr_code/testing/`, expanded `TestOutcome`, unit tests, `scripts/test_attempts.py`, and `scripts/demo_stage3.py`. dr-queues handler wiring deferred to pipeline phase. Next: [Stage 4 — Analysis](./stage-04-analysis.md) (complete); pipeline phase is next — see [overview](./overview.md#implementation-phasing-suggested).

## Purpose

Run **HumanEval+ functional tests** on code extracted in stage 2. Produce standardized **test outcomes** suitable for granular analysis (per-test-case pass/fail, not only a single boolean).

Stage 3 is the **eval bottleneck** — Docker batch execution, queue-parallel from v1.

---

## Dependency

**nl-code** for Docker-isolated execution:

- `HumanEvalDataset` / `RawHumanEvalTask` **or** dr-code task model + nl-code test parsing (prefer reusing nl-code test execution paths even if stage 1 loader is local).
- `run_test_cases` / batch runners from `nl_code.code_execution.runner`.
- Pattern from `nl_code.optim.humaneval_dspy_eval.evaluate_completed_code`: wrap extracted code with entry-point helper, run cases.

Install as path dependency on `../nl-code` (with `[docker]` extra).

---

## Output shape: `TestOutcome`

```text
TestOutcome
├── sample_id, run_id, task_id
├── parse_success                  # copied from stage 2
├── skipped                        # true if parse failed or policy skip
├── skip_reason                    # e.g. "parse_failed", "infra_error"
│
├── entry_point
├── extracted_code                 # code that was tested
│
├── test_pass_rate                 # float 0..1
├── all_tests_passed               # test_pass_rate == 1.0
├── test_case_results[]            # per case: input, expected, actual, passed, error
│
├── infra_error                    # Docker/worker failure (distinct from test fail)
└── latency_ms / batch_id          # optional
```

Align field names with nl-code `TestCaseResult` where possible to avoid lossy mapping.

---

## Handler behavior

Per job arriving from parse stage:

1. If `parse_success` is false → emit `TestOutcome` with `skipped=true`, no Docker.
2. Else build eval code from `extracted_code` + `entry_point` (nl-code helper pattern).
3. Load test cases for `task_id`.
4. **Docker batch:** spin up container → run batch of test cases for this job (or N jobs batched — see open questions) → tear down container.
5. Project results to `TestOutcome`.
6. Persist to Mongo (see storage open question).
7. Terminal event / job completion.

**Worker lifecycle (solidified direction):**

- Worker process pulls jobs from test queue.
- For each batch unit: start Docker worker runtime → execute nl-code batch API → stop container.
- Avoid long-lived containers across unrelated tasks to limit state leaks; amortize startup via batch size tuning.

---

## Queue integration

**Pipeline position:** parse stage completed queue → **test workers** → terminal.

dr-queues configuration:

- Separate worker pool sizing for parse vs test (many parse workers, fewer test workers bounded by Docker/CPU).
- Manifest records `workers_by_stage: { parse: N, test: M }`.
- Detached workers: `dr-queues-stage-worker --stage test --handlers-module dr_code.pipeline.handlers`.

**Seeding strategy (solidified direction):**

- Prefer dedup-aware jobs: one eval job per unique `(task_id, raw_output)` with `occurrence_count`.
- Optional: batch multiple samples per Docker invocation within one handler (same task only if tests differ — usually one task per batch unit).

---

## MongoDB storage

dr-queues defaults to `pipeline_events` collection for lifecycle telemetry.

**Solidified direction:** eval needs queryable **result documents** for stage 4. Likely a dedicated collection (e.g. `eval_results`) keyed by `(run_id, sample_id)` in addition to pipeline events — exact layout is an open question.

Requirements:

- Idempotent upsert on retry
- Index on `run_id`, `task_id`, `all_tests_passed`, `provenance.source`
- Store enough of `AttemptRecord` provenance for slicing without rejoining stage 1 export

---

## Infrastructure

- Docker image: nl-code scientific eval image (see nl-code README).
- Local dev: RabbitMQ + Mongo via docker compose (dr-queues convention).
- Environment: `AMQP_URL`, `MONGODB_URL`, Docker socket available to test workers.

---

## Testing strategy (for implementers)

1. Smoke: single task, single sample, in-process pipeline before detached workers.
2. `@pytest.mark.docker` integration tests for handler with known pass/fail code.
3. Scale test: one task dedup file end-to-end with small worker counts.

---

## Solidified design points

- nl-code owns Docker execution; dr-code owns orchestration and outcome schema.
- Test stage is queue-backed from v1 with Docker batch workers.
- Parse failures skip Docker explicitly with recorded reason.
- Separate infrastructure errors from test failures (nl-code contract).
- Dedup-aware seeding to reduce Docker cost on pool replay.

---

## Open questions

- **Results collection schema:** one document per sample vs embedded arrays in run document?
- **Batch size:** jobs per container invocation (1 vs N samples); same-task constraint?
- **Task test cache:** load all HumanEval+ tests in worker memory vs per-job lookup?
- **Partial run resume:** re-seed only missing `(run_id, sample_id)` — manifest tracking?
- **Worker placement:** one Docker per worker process vs pool of containers?
- **nl-code import surface:** depend on full package vs minimal execution submodule?
