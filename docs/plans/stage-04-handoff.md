# Stage 4 handoff — Analysis

Brief context for the agent implementing Stage 4. Read the full design in [stage-04-analysis.md](./stage-04-analysis.md) and [overview.md](./overview.md) first.

---

## Your mission

Build the **offline analysis layer** that joins compression metrics on decoder inputs with test outcomes from stage 3, sliced by experiment metadata (model, source, pool, template, etc.). Stage 4 scope is a deterministic Typer CLI plus a marimo exploration notebook — not live queue/Mongo orchestration (that remains the **pipeline** phase between stages 3 and 4 at scale).

---

## What Stage 3 left you

### Input contracts

**`TestOutcome`** — defined in [`src/dr_code/models/outcomes.py`](../../src/dr_code/models/outcomes.py). Key fields for analysis:

| Field | Notes |
|-------|-------|
| `sample_id`, `run_id`, `task_id` | Join keys — match stage 1 `AttemptRecord` and stage 2 `ParseOutcome` |
| `outcome_kind` | `skipped` \| `tested` \| `infra_error` \| `internal_error` — primary classifier |
| `tests_ran` | `true` only when Docker completed; per-case results trustworthy only then |
| `all_tests_passed`, `test_pass_rate` | Primary correctness metrics when `outcome_kind=tested` |
| `test_case_results[]` | Per-case pass/fail when `tests_ran=true` |
| `parse_success` | Copied from stage 2 — funnel analysis |
| `skipped`, `skip_reason` | Parse/policy skip (no Docker) |
| `infra_error`, `internal_error` | Structured failure — **exclude or slice separately** from correctness stats |

**Upstream join columns** still live on `AttemptRecord` (not fully denormalized onto `TestOutcome` today):

| Field | Analysis use |
|-------|----------------|
| `decoder_input` | zstd22 compression target |
| `provenance.source` | `pool` vs `fresh_stub` — never merge without labeling |
| `provenance.model`, `pool_name`, `prompt_template_id`, … | Slice dimensions |
| `provenance.occurrence_count` | Weighted aggregates for deduped pool rows |

Also available: `ParseOutcome` (`parse_success`, `code_eval` provenance) via join on `sample_id`.

### Ready-made artifacts to analyze locally

After stages 1–3 demos:

```bash
uv run scripts/demo_stage1.py --skip-live
uv run scripts/parse_attempts.py \
  --input exports/demo/pool.parquet \
  --output exports/demo/parse.jsonl
uv run scripts/test_attempts.py \
  --attempts exports/demo/pool.parquet \
  --parse exports/demo/parse.jsonl \
  --output exports/demo/test.jsonl
```

| Path | Contents |
|------|----------|
| `exports/demo/pool.parquet` | 7 pool `AttemptRecord` rows |
| `exports/demo/fresh.parquet` | 1 fresh_stub row |
| `exports/demo/parse.jsonl` | `ParseOutcome` per line |
| `exports/demo/test.jsonl` | `TestOutcome` per line (regenerate after test changes) |

Join attempts → parse → test on `sample_id`. For a full run export, also carry `run_id`.

### Test adapter you call upstream (or receive from queue)

[`src/dr_code/testing/adapter.py`](../../src/dr_code/testing/adapter.py):

```python
from dr_code.testing import test_parsed_sample

outcome = test_parsed_sample(record, parse_outcome)
```

Demos:

- `uv run scripts/demo_stage3.py` — stages 1–3 walkthrough on one sample with outcome banners
- `uv run scripts/demo_stage3.py --show-failure --show-canonical`
- `uv run scripts/test_attempts.py --help` — batch test export (resilient per-row containment)

---

## Stage 3 design decisions (already resolved)

These affect how stage 4 should treat outcomes:

1. **`outcome_kind` is authoritative.** Do not infer infra from `test_pass_rate=0`. Filter `outcome_kind=tested` for correctness; report `infra_error` / `internal_error` separately in summaries.

2. **Per-case results only when `tests_ran=true`.** Empty `test_case_results` on infra/internal/skip — not "all failed."

3. **One Docker container per sample** in v1 (`run_test_cases`). Batch CLI loops samples; it does not batch containers.

4. **Parse success ≠ correct solution.** Pool rows often parse to stubs like `return False`. Expect mixed pass rates — that is real signal for compression–correctness analysis.

5. **Weighted stats.** Use `occurrence_count` from pool provenance when aggregating deduped rows.

6. **Forward on parse fail.** Test stage always runs; `outcome_kind=skipped` with explicit reason. Include in parse→test funnel views.

---

## Suggested implementation shape

```text
src/dr_code/analysis/
  compress.py       # zstd22 on decoder_input (match dr-bottleneck convention)
  join.py           # AttemptRecord + ParseOutcome + TestOutcome → enriched row
  aggregate.py      # pass rate by model/source/task; weighted counts
  export.py         # Parquet/CSV/JSON summary writers
scripts/analyze_eval_run.py   # typer CLI
nbs/analyze_eval_run.py       # marimo exploration (loads script exports)
```

1. **Row-level enrich:** join on `(run_id, sample_id)`; add `decoder_input_len_raw`, `decoder_input_len_zstd22`, test flags, provenance columns.
2. **Aggregates:** pass rate by model, source, task; compression quartile × pass; parse funnel (`raw → parse_success → tested → all_tests_passed`).
3. **Summary JSON:** headline counts including `outcome_kind` breakdown (skipped / tested+pass / tested+fail / infra / internal).
4. **Marimo notebook:** charts only — script exports are source of truth for metrics.
5. **Join failure report:** samples in attempts export but missing test outcome — count and list in summary.

---

## Dependencies

| Dep | Status | Stage 4 usage |
|-----|--------|---------------|
| `zstandard` or stdlib | **Not wired yet** | zstd level 22 on `decoder_input` |
| Stage 3 exports | **Available locally** | `test.jsonl` + attempt parquet + optional `parse.jsonl` |
| Mongo `eval_results` | **Not wired yet** | Optional `--mongodb-url` later; v1 can be export-first |

Pipeline phase (before large-scale stage 4): wire dr-queues parse → test handlers, Mongo sink, seed CLI — see overview phasing bullets 6–8. Stage 4 v1 can run entirely from JSONL/Parquet exports produced by local CLIs.

---

## Verification targets

```bash
uv run pytest tests/unit -q
# After analysis module exists:
uv run pytest tests/unit/test_analysis_*.py -q
uv run scripts/analyze_eval_run.py \
  --attempts exports/demo/pool.parquet \
  --parse exports/demo/parse.jsonl \
  --test exports/demo/test.jsonl \
  --output-dir exports/demo/analysis
uv run marimo run nbs/analyze_eval_run.py
```

Success criteria for stage 4 v1 (export-first):

- Enriched row table includes zstd22 size + `all_tests_passed` + provenance slices
- Aggregates separate `pool` vs `fresh_stub` and weight by `occurrence_count`
- Summary reports join failures and `outcome_kind` counts (infra not counted as test fail)
- Marimo notebook loads enriched Parquet and renders compression vs pass chart

---

## Open decisions (resolve during stage 4)

From [stage-04-analysis.md](./stage-04-analysis.md):

- **Export-first vs live Mongo query:** recommend Parquet snapshot default for notebook reproducibility
- **Binning:** fixed byte buckets vs per-run quantiles for compression charts
- **Comparison runs:** convention for two `run_id`s in one notebook
- **Joint objective preview:** document `pass - λ * compressed_len` scalar for future DSPy — do not optimize yet

---

## Docs & plans

- Design: [stage-04-analysis.md](./stage-04-analysis.md)
- Stage 3 (complete): [stage-03-testing.md](./stage-03-testing.md)
- Stage 2 (complete): [stage-02-parsing.md](./stage-02-parsing.md)
- dr-bottleneck compression convention: [../investigation/dr-bottleneck.md](../investigation/dr-bottleneck.md)
