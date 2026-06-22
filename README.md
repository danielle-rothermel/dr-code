# dr-code

Research harness for the compression–correctness evaluation pipeline: given a natural-language description of a HumanEval function, can a decoder reconstruct working Python, and how compressible is the description?

Design docs: [docs/plans/README.md](docs/plans/README.md)

**Status:** Stages 1–4 complete. Pipeline (dr-queues parse→test) wired — see [pipeline runbook](docs/plans/pipeline-runbook.md).

## Pipeline demo

In-process smoke on a few pool dedup samples (requires RabbitMQ and Mongo):

```bash
cd ../dr-queues && docker compose up -d
uv run scripts/demo_pipeline.py --limit 3
```

## Pipeline proof / batch run

Detached parallel eval on pool dump artifacts:

```bash
uv run scripts/run_eval_pipeline.py \
  --mode detached \
  --task-indices 0,1,2,3,4 \
  --workers parse=8,test=8
```

Outputs: `exports/runs/{run_id}/` (attempts, parse/test JSONL, `proof_report.json`). Tune test workers mid-run with `scripts/tune_test_workers.py`. Full runbook: [docs/plans/pipeline-runbook.md](docs/plans/pipeline-runbook.md).

## Stage 4 demo

Join attempt, parse, and test exports; write enriched Parquet, summary JSON, and aggregate tables. Explore in marimo.

```bash
uv run scripts/analyze_eval_run.py \
  --attempts exports/demo/pool.parquet \
  --parse exports/demo/parse.jsonl \
  --test exports/demo/test.jsonl \
  --output-dir exports/demo/analysis

uv run marimo run nbs/analyze_eval_run.py
```

## Stage 3 demo

Batch-test a parse export with local fork workers:

```bash
uv run scripts/test_attempts.py \
  --attempts exports/demo/pool.parquet \
  --parse exports/demo/parse.jsonl \
  --output exports/demo/test.jsonl

uv run scripts/demo_stage3.py --show-failure
```

## Stage 2 demo

Walk one `AttemptRecord` through code-eval parsing: raw output, extracted code, provenance, and mongosh inspect commands for when the pipeline is wired.

```bash
uv run scripts/demo_stage2.py --show-failure
```

Batch-parse an export to JSONL:

```bash
uv run scripts/parse_attempts.py \
  --input exports/demo/pool.parquet \
  --output exports/demo/parse.jsonl
```

## Stage 1 demo

Full verification: pool import, live fresh generation (or offline stub), export, stats, and side-by-side spot check.

```bash
# Offline smoke (no API key)
uv run scripts/demo_stage1.py --skip-live

# Live generation (requires OPENROUTER_API_KEY)
uv run scripts/demo_stage1.py
```

Exports land in `exports/demo/pool.parquet` and `exports/demo/fresh.parquet`.

## Stage 1 CLIs

Import pool artifacts into unified `AttemptRecord` exports:

```bash
uv run scripts/import_pool_attempts.py --help
```

Generate fresh decoder attempts via dr-providers:

```bash
uv run scripts/generate_decoder_attempts.py --list-profiles
uv run scripts/generate_decoder_attempts.py \
  --profile openrouter/google/gemini-3.1-flash-lite/off/v1 \
  --task-ids HumanEval/0 --stats
```

Rebuild the offline HumanEval+ snapshot (requires network):

```bash
uv run scripts/build_humaneval_snapshot.py
```

## Tests

```bash
uv run pytest tests/unit -q
```
