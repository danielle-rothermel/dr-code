# Stage 4 — Analysis

[← Overview](./overview.md) · **Status:** Done (2026-06-21). Module in `src/dr_code/analysis/`, unit tests, `scripts/analyze_eval_run.py`, and `nbs/analyze_eval_run.py`. Export-first v1; Mongo query deferred to pipeline phase. Next: [pipeline phase](./overview.md#implementation-phasing-suggested).

## Purpose

Join **compression metrics** on decoder inputs with **test outcomes** from stage 3, sliced by experiment metadata (model, source, pool, template, etc.). Support exploratory visualization in marimo.

Analysis is **offline** — runs against completed Mongo exports or snapshots, not live queues.

---

## Inputs

| Input | Source |
|-------|--------|
| Decoder input text | `AttemptRecord.decoder_input` (from stage 1 fields preserved in eval results) |
| Test outcomes | Stage 3 Mongo `eval_results` (or export) |
| Provenance dimensions | `provenance.*` on each record |
| Occurrence weights | `occurrence_count` for deduped pool rows |

Optional: re-import stage 1 Parquet for columns not copied into eval results (avoid if eval store is complete).

---

## Core metrics

### Compression (solidified)

Apply **zstd level 22** to `decoder_input` (UTF-8 byte length before/after or compressed size only — match dr-bottleneck convention):

```text
decoder_input_len_raw       = len(decoder_input.encode("utf-8"))
decoder_input_len_zstd22    = len(zstd_compress(decoder_input, level=22))
```

Rationale: encoder optimization target is description size vs correctness; pool experiments used encode-side compression in dr-bottleneck — stage 4 applies the same metric to the **description the decoder actually saw**.

Optional later: also measure `raw_output` / `extracted_code` sizes for debugging.

### Correctness

From `TestOutcome`:

- `all_tests_passed` (primary binary metric)
- `test_pass_rate` (partial credit)
- Per-test-case breakdown for failure mode charts

### Parse layer (optional slices)

From `ParseOutcome`:

- `parse_success` rate by source/model
- Recovery attribution summaries (which repairs fired) — secondary analysis

---

## Outputs

### Bulk processing script

Deterministic CLI `scripts/analyze_eval_run.py`:

- Args: `--attempts`, `--parse`, `--test`, `--output-dir` (optional `--limit`)
- Produces:
  - `enriched.parquet` — row-level table with zstd metrics + test flags + provenance
  - `summary.json` — headline numbers, parse funnel, join failures, joint-objective preview
  - `aggregates/*.parquet` — pass rate by source, model, task, compression quartile

### Marimo notebook

`nbs/analyze_eval_run.py`:

- Loads `exports/demo/analysis/` (or any `--output-dir` from the CLI)
- Charts: compression vs pass scatter, parse funnel, source comparison, task hardness table
- Not the source of truth for metrics — script exports are

---

## Typical analysis views

| View | Dimensions |
|------|------------|
| Compression vs pass | `decoder_input_len_zstd22` × `all_tests_passed`, color by model |
| Source comparison | `pool` vs `fresh_stub` (never merge without labeling) |
| Model leaderboard | pass rate weighted by `occurrence_count` |
| Parse funnel | raw → parse_success → test_pass |
| Task hardness | pass rate by `task_id` |

---

## Dependency on earlier stages

- Requires stable join keys: `run_id`, `sample_id`, `task_id`.
- Stage 3 documents must retain `decoder_input` (or joinable reference).
- Stage 1 `provenance.source` must survive into eval store for pool vs fresh_stub splits.

---

## Solidified design points

- zstd22 on **decoder input** text (not encode output from a separate column unless added later).
- Script for reproducible aggregates; marimo for exploration.
- Weighted stats using `occurrence_count` when present.
- Slice by all provenance dimensions captured in stage 1.

---

## Resolved decisions (v1)

- **Export-first vs query Mongo live:** Parquet/JSONL exports are the default; Mongo `--mongodb-url` deferred.
- **Binning strategy:** per-run quantiles (Q1–Q4 on `decoder_input_len_zstd22`), stored on enriched rows.
- **Joint objective preview:** documented in `summary.json` as `pass - λ * compressed_len`; not optimized.
- **Join failures:** reported in `summary.json` (`join_failures.missing_test_sample_ids`).
- **Comparison runs:** single-run v1; multi-run comparison loads multiple enriched Parquet files (documented in summary schema note).

## Open questions (deferred)

- Fixed byte buckets as an alternative to per-run quantile binning
- Side-by-side multi-`run_id` comparison UI in marimo

---

## Future: DSPy optimization feedback

When encoder optimization lands, stage 4 metrics become the **reporting layer** for training/eval splits:

- Same analysis script keyed by optimization trial id
- Add encoder prompt version / DSPy program hash to provenance dimensions
- Per-test-case breakdown charts in marimo (failure mode inspection)
- Mongo live query path when pipeline writes `eval_results`

Schema leaves room in `provenance.extra` for additional optimization dimensions.
