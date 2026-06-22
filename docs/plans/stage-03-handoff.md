# Stage 3 handoff — Testing

Brief context for the agent implementing Stage 3. Read the full design in [stage-03-testing.md](./stage-03-testing.md) and [overview.md](./overview.md) first.

---

## Your mission

Build the **nl-code test adapter** that turns each parsed sample into a `TestOutcome`: run HumanEval+ functional tests on `ParseOutcome.extracted_code`, or emit an explicit skip when parsing failed. Stage 3 scope is the testing logic + unit tests (and optional local demo/CLI); full queue/Mongo orchestration is the combined **stage 2–3 pipeline** phase (see phasing in overview), but the adapter must be queue-ready.

---

## What Stage 2 left you

### Input contract: `ParseOutcome` (+ upstream `AttemptRecord`)

`ParseOutcome` is defined in [`src/dr_code/models/outcomes.py`](../../src/dr_code/models/outcomes.py). Key fields for testing:

| Field | Notes |
|-------|-------|
| `sample_id`, `run_id`, `task_id` | Join keys — carry through to `TestOutcome` |
| `parse_success` | If `false`, skip Docker; emit `TestOutcome.skipped=true` |
| `extracted_code` | Python to execute when `parse_success=true` |
| `skip_reason` | e.g. `"no_valid_candidate"` when parse failed |
| `code_eval` | Parse provenance only; not needed for test execution |

You also need **`entry_point`** from the original `AttemptRecord` (not stored on `ParseOutcome` today). Load from export or pass through job payload alongside parse output.

`TestOutcome` in the same file is a **skeleton** — expand it to match [stage-03-testing.md](./stage-03-testing.md) (`test_case_results`, `infra_error`, `entry_point`, `extracted_code`, etc.).

### Ready-made artifacts to test against

After stage 1 + 2 demos:

```bash
uv run scripts/demo_stage1.py --skip-live   # AttemptRecord exports
uv run scripts/parse_attempts.py \
  --input exports/demo/pool.parquet \
  --output exports/demo/parse.jsonl
```

| Path | Contents |
|------|----------|
| `exports/demo/pool.parquet` | 7 pool `AttemptRecord` rows |
| `exports/demo/fresh.parquet` | 1 fresh_stub row |
| `exports/demo/parse.jsonl` | `ParseOutcome` per line (regenerate after parse changes) |

Join parse outcomes to attempts on `sample_id` when building test inputs locally.

HumanEval+ tasks (including **`test`** string for execution) load from snapshot:

```python
from dr_code.datasets.humaneval_loader import get_task
task = get_task("HumanEval/0", prefer_snapshot=True)
# task.entry_point, task.test
```

Snapshot: `tests/corpus/humanevalplus_snapshot.json` (164 tasks, offline).

### Parsing adapter you call upstream (or receive from queue)

[`src/dr_code/parsing/adapter.py`](../../src/dr_code/parsing/adapter.py):

```python
from dr_code.parsing import parse_attempt

outcome = parse_attempt(record)  # EXTRACTION_CONFIG, best_valid_source()
```

Demos:

- `uv run scripts/demo_stage2.py --show-failure` — parse walkthrough + mongosh inspect hints
- `uv run scripts/parse_attempts.py --help` — batch parse export

---

## Dependencies to wire

| Dep | Status | Stage 3 usage |
|-----|--------|---------------|
| `code-eval==0.1.1` | Editable `../code-eval` | Stage 2 only; test stage consumes `ParseOutcome` |
| `nl-code` | **Not wired yet** | Docker test execution — `run_test_cases`, batch runners |
| `dr-queues` | **Not wired yet** | Parse → test pipeline orchestration (with stage 2 handler) |

**Install nl-code** as editable path dep on `../nl-code` (likely with `[docker]` extra). Investigation notes: [../investigation/nl-code.md](../investigation/nl-code.md).

**Execution pattern to reuse:** `nl_code.optim.humaneval_dspy_eval.evaluate_completed_code` — wrap extracted code with entry-point helper, run per-case tests. Prefer nl-code's execution paths over reimplementing HumanEval `check()`.

**HumanEval+ loader:** dr-code owns a lightweight task model ([`HumanEvalPlusTask`](../../src/dr_code/models/humaneval.py)); use `task.test` from snapshot loader rather than pulling nl-code's full dataset layer for stage 3 v1.

---

## Suggested implementation shape

```text
src/dr_code/testing/
  adapter.py      # test_parsed_sample(record, parse_outcome) -> TestOutcome
  config.py       # execution mode / Docker image helpers (optional)
```

1. **`test_parsed_sample(record, parse_outcome) -> TestOutcome`** (or accept `AttemptRecord` + `ParseOutcome`) — thin wrapper around nl-code execution.
2. **Parse-fail path:** `parse_success=false` → `skipped=true`, `skip_reason="parse_failed"` (or propagate parse `skip_reason`), no Docker.
3. **Success path:** build eval code from `extracted_code` + `entry_point`, load `task.test`, run cases in Docker via nl-code.
4. **Unit tests** without Docker where possible (skip-path projection, code wrapping); `@pytest.mark.docker` for smoke integration.
5. **Optional local CLI** — e.g. `scripts/test_attempts.py` reading parse JSONL + attempt export; **`scripts/demo_stage3.py`** for single-sample before/after (mirror stage 2 demo).

Queue handler (later, with dr-queues): read `ParseOutcome` from `step_outputs["parse"]` + attempt fields → call adapter → write `step_outputs["test"]` → terminal event / `eval_results` upsert.

---

## Things we learned in Stage 2

1. **Parse success ≠ correct solution.** code-eval validates syntax/shape only. Pool rows often parse to stubs like `return False`. Expect mixed pass rates in stage 3 — that's real signal.

2. **`extraction_log_summary` `valid=` flags** reflect which extractors contributed valid *candidates* (after code-eval backfills `yielded_valid_candidate`). `valid_count` is total validated candidates, not log line count.

3. **Slim projections.** Store `TestOutcome` in Mongo/eval exports, not full nl-code raw payloads, unless debugging.

4. **Forward on parse fail (solidified).** Test handler always runs; records skip explicitly. Keeps pipeline shape uniform for dr-queues.

5. **Dedup semantics unchanged.** Same `sample_id` for identical `(task_id, raw_output)`; test once per unique sample at scale; carry `occurrence_count` from `AttemptRecord.provenance` into result docs for weighted stage 4 analysis.

6. **Stage 2 demo mongosh commands** in `demo_stage2.py` show future `eval_results` shape — stage 3 should populate those documents when pipeline lands.

---

## Verification targets

```bash
uv run pytest tests/unit -q
# After adapter exists:
uv run pytest tests/unit/test_testing_adapter.py -q
uv run pytest tests/unit/test_testing_adapter.py -q -m docker   # if marked
# Optional single-sample demo:
uv run scripts/demo_stage3.py
# Optional batch:
uv run scripts/test_attempts.py \
  --attempts exports/demo/pool.parquet \
  --parse exports/demo/parse.jsonl \
  --output exports/demo/test.jsonl
```

Success criteria for stage 3 adapter (before queue):

- Known-good extracted code (canonical solution body) → `all_tests_passed=true`
- Parse-fail row → `skipped=true`, no Docker invocation
- Known-bad extracted code → tests run, `all_tests_passed=false` (not infra error)
- Infrastructure failures surfaced as `infra_error` distinct from test failure

**Prerequisites for Docker tests:** Docker daemon running, nl-code eval image available (see nl-code README).

---

## Open decisions (resolve during stage 3)

From [stage-03-testing.md](./stage-03-testing.md):

- **`TestOutcome` field set:** align with nl-code `TestCaseResult`; include per-case array vs summary-only for v1?
- **Mongo `eval_results` schema:** one doc per `(run_id, sample_id)` vs nested run document?
- **Batch size:** one sample per Docker invocation vs mini-batch in handler?
- **nl-code import surface:** full package vs minimal execution submodule?

Recommend: **one doc per sample** in `eval_results`; **one sample per handler call** for v1 (batch tuning later); **full nl-code path dep with `[docker]`** for correctness; **include `test_case_results[]`** for stage 4 granularity.

Pipeline phase (after adapter): wire `dr-queues` parse → test handlers, Mongo sink, seed CLI — see overview phasing bullets 5–6.

---

## Docs & plans

- Design: [stage-03-testing.md](./stage-03-testing.md)
- Stage 2 (complete): [stage-02-parsing.md](./stage-02-parsing.md)
- nl-code investigation: [../investigation/nl-code.md](../investigation/nl-code.md)
- dr-queues conventions: [../dr-queues README](https://github.com/danielle-rothermel/dr-queues) (local `../dr-queues`)
