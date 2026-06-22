# Pipeline runbook

dr-code eval pipeline: dr-queues parse → test on pool dump artifacts.

## Infrastructure

Start RabbitMQ and Mongo (from sibling repo):

```bash
cd ../dr-queues && docker compose up -d
```

Defaults:

- `AMQP_URL=amqp://guest:guest@localhost:5672/`
- `MONGODB_URL=mongodb://localhost:27017/dr_queues`

Pre-flight (automatic in CLIs, or manual):

```bash
uv run scripts/eval_run.py preflight
```

Requires:

- RabbitMQ and Mongo reachable
- Pool dump at `DEFAULT_DUMP_DIR` (see `src/dr_code/pipeline/seed.py`)

## Simple demo (in-process smoke)

1–3 dedup samples from fixtures:

```bash
uv run scripts/demo_pipeline.py --limit 3
```

## Beefy driver

### In-process (debug / medium runs)

```bash
uv run scripts/eval_run.py run \
  --mode in-process \
  --task-indices 0 \
  --limit-per-task 10 \
  --workers parse=4,test=1
```

### Detached (parallel proof / production)

Recommended on Mac Mini after tuning (`proof-20840125`):

```bash
uv run scripts/eval_run.py run \
  --mode detached \
  --task-indices 0,1,2,3,4 \
  --workers parse=8,test=8 \
  --completion-timeout 28800
```

For all remaining pool tasks, expand `--task-indices` or pass the full index list.

## Outputs

Each run writes under `exports/runs/{run_id}/`:

| File | Contents |
|------|----------|
| `attempts.parquet` | Seeded AttemptRecord rows |
| `parse.jsonl` | ParseOutcome per line |
| `test.jsonl` | TestOutcome per line |
| `manifest.json` | dr-queues run manifest copy |
| `proof_report.json` | Timing + outcome summary |
| `tune_report.json` | Live worker sweep results (when tuned) |
| `analysis/` | Stage 4 enriched Parquet + summary (after analyze CLI) |

Mongo collections:

- `pipeline_events` — dr-queues lifecycle telemetry
- `eval_results` — upserted TestOutcome documents keyed by `(run_id, sample_id)`

Run manifest: `.runs/{run_id}/manifest.json`

## Proof bar acceptance (HumanEval/0–4)

1. `terminals == expected_jobs` (~5,828 dedup rows for indices 0–4)
2. `proof_report.json` shows per-stage and per-task samples/sec
3. Outcome sanity: `outcome_kind` breakdown; infra errors sliced separately
4. Stage 4 capstone:

```bash
uv run scripts/analyze_eval_run.py \
  --attempts exports/runs/{run_id}/attempts.parquet \
  --parse exports/runs/{run_id}/parse.jsonl \
  --test exports/runs/{run_id}/test.jsonl \
  --output-dir exports/runs/{run_id}/analysis
```

5. Zero join failures in analysis `summary.json`

### Proof run `proof-20840125` (2026-06-21)

| Metric | Value |
|--------|-------|
| Jobs | 5,828 / 5,828 |
| Wall time | 994 s (~16.6 min) |
| Overall throughput | 5.86 samples/sec |
| Parse throughput | ~38 samples/sec |
| Test throughput (mixed workers) | ~1.7 samples/sec |
| Tested pass rate | 25.3% (26.6% weighted) |
| Outcomes | 5,810 tested, 13 skipped, 5 infra_error |
| Join failures | 0 |

Proof used mixed test worker counts (started at 2, tuned to 8 mid-run). Use `test=8` from the start for full pool runs.

## Live test-worker tuning

While a detached run is in **test-only phase** (parse complete, terminals < expected), hot-swap test workers and measure throughput:

```bash
uv run scripts/tune_test_workers.py \
  --run-id YOUR_RUN_ID \
  --start-workers 2 \
  --window-seconds 60 \
  --warmup-seconds 15 \
  --max-workers 16
```

### Algorithm

1. Measure baseline at `start_workers` (no swap if already running).
2. Double workers (`×2`) each step: swap via `dr-queues-stage-worker --replace`.
3. **Warmup** 15 s after each swap (in-flight jobs requeue).
4. **Measure** 60 s — poll Mongo `terminal` event count.
5. **Stop** when throughput drops (regression) or gain vs previous step is < 10%.
6. **Apply best** — leave winning worker count running (`--apply-best`, default).

Also stops parse workers automatically once parse stage completes (frees CPU).

Writes `exports/runs/{run_id}/tune_report.json`. Does **not** stop the main `scripts/eval_run.py run` driver.

### Mac Mini results (`proof-20840125`)

| test workers | samples/sec | delta vs prior |
|--------------|-------------|----------------|
| 2 | 5.08 | baseline |
| 4 | 6.45 | +27% |
| 8 | **6.60** | +2.3% (below 10% threshold → stop) |

**Recommendation:** `--workers parse=8,test=8` for production pool runs on this machine. Infra errors during sweep: +1 at 4 workers, +4 at 8 workers (5 total across full proof — acceptable).

Monitor during tuning:

```bash
watch -n 10 'mongosh mongodb://localhost:27017/dr_queues --quiet --eval "db.pipeline_events.countDocuments({run_id:\"YOUR_RUN_ID\", event:\"terminal\"})"'
```

## Detached workers (manual)

```bash
uv run scripts/eval_run.py start \
  --run-id YOUR_RUN_ID \
  --stage test \
  --workers parse=0,test=8
```

Parse workers can be stopped manually after parse completes (optional — tune script does this).

## Dump path

Default pool dump:

`/Users/daniellerothermel/drotherm/data/code-comp/dr-llm-humaneval-pool-dumps/20260621_manual`

Per-task artifacts: `per_elem/human_eval-{n}-decode-dedup.jsonl` + `human_eval-{n}-decode.parquet`

Override with `--dump-dir`.

Scale: ~172k dedup unique raw strings across 163 tasks; indices 0–4 alone are 5,828 rows.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Jobs requeued forever | Handler exception — check worker stderr; parse handler should catch code-eval throws |
| Terminal count stall | Test workers alive; raise `--completion-timeout` |
| High `infra_error` | Reduce test workers; inspect worker stderr and fork timeout stages |
| RabbitMQ port conflict | Another broker on 5672 — existing instance OK if reachable |
| Log quiet after parse burst | Normal — only test stage logs once parse finishes |
| Oversized pool sample | Parse records failure (`parse_handler_error`); does not infinite-loop |
