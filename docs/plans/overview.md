# dr-code — Pipeline overview

## Mission

**dr-code** is the research harness that standardizes evaluation for the compression–correctness question: given a natural-language description of a HumanEval function, can a decoder reconstruct **working** Python, and how compressible is the description?

It connects historical and fresh decoder outputs to:

1. **Parsing** — recover valid Python from messy LLM text (code-eval)
2. **Testing** — run HumanEval+ cases in local fork workers
3. **Analysis** — relate description compression (zstd22) to test outcomes, sliced by experiment metadata

Ultimate downstream goal (not in initial scope): **DSPy optimization of encoder prompts** for the joint compression + correctness objective. dr-bottleneck will later adopt the same stage contracts for large-scale enc/dec runs.

Background: [Investigation synthesis](../investigation/synthesis.md). Operations: [Pipeline runbook](./pipeline-runbook.md).

**Status (2026-06-21):** Stages 1–4 and the dr-queues pipeline are implemented and verified. Proof bar passed on HumanEval/0–4 (`proof-20840125`, 5,828 dedup rows).

---

## End-to-end flow

```text
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1 — Generation dataset                                     │
│  (a) dr-llm pool extract  OR  (b) HumanEval+ + dr-providers      │
│  → unified AttemptRecord rows                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stages 2–3 — Eval pipeline (dr-queues)                           │
│  parse queue → code-eval workers (in-process)                    │
│  test queue  → local fork workers (one child process per sample)   │
│  → pipeline_events + eval_results (Mongo) + file exports         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4 — Analysis (offline)                                     │
│  zstd22(decoder_input) ⨝ test outcomes → enriched Parquet/JSON   │
│  marimo notebook for exploration                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 1 — Generation & dataset

Produce unified **AttemptRecord** rows from two sources:

| Source | `provenance.source` | Input |
|--------|---------------------|-------|
| **1a Pool replay** | `pool` | dr-llm HumanEval pool extract (Parquet / dedup JSONL) |
| **1b Fresh generation** | `fresh_stub` | HumanEval+ tasks + dr-providers batch decoder calls |

Future: `fresh_encoded` — real encoder output at a budget (for DSPy and pool-comparable runs).

**Key fields:** `sample_id` (SHA-256 of `task_id` + `raw_output`), `decoder_input`, `raw_output`, `task_id`, `entry_point`, `provenance.*` including `occurrence_count` for dedup rows.

**Pool artifacts** (external): see [dr-llm pool investigation](../investigation/dr-llm-humaneval-pool.md). Prefer dedup JSONL + Parquet join for pipeline seeding.

**Scripts:** `import_pool_attempts.py`, `generate_decoder_attempts.py`, `demo_stage1.py`, `build_humaneval_snapshot.py`

**Modules:** `dr_code.datasets.*`, `dr_code.generation.*`

---

## Stage 2 — Parsing

Turn `raw_output` into **ParseOutcome** via code-eval `EXTRACTION_CONFIG` (no subprocess normalizers at pool scale).

- Call `LLMCodeValidator.validate()` → project with `best_valid_source()` / `best_valid_candidate()`
- Parse handler catches exceptions (e.g. oversized samples) and records `parse_success=false` instead of requeue loops
- Always forward to test stage; test emits explicit skip when parse failed

**Scripts:** `parse_attempts.py`, `demo_stage2.py`  
**Modules:** `dr_code.parsing.*`  
**Pipeline handler:** `dr_code.pipeline.handlers.parse_attempt`

---

## Stage 3 — Testing

Run HumanEval+ functional tests on extracted code via local fork execution.

**TestOutcome** uses authoritative `outcome_kind`: `tested` | `skipped` | `infra_error` | `internal_error`. Per-case results only when `tests_ran=true`. v1: one forked child process per sample.

**Scripts:** `test_attempts.py`, `demo_stage3.py`  
**Modules:** `dr_code.testing.*`  
**Pipeline handler:** `dr_code.pipeline.handlers.run_tests`

---

## Stage 4 — Analysis

Offline joins on export files (or Mongo snapshots):

- **Compression:** zstd level 22 on `decoder_input` (match dr-bottleneck convention)
- **Correctness:** `all_tests_passed`, `test_pass_rate`, weighted by `occurrence_count`
- **Slices:** source, model, task, compression quartile; parse funnel

**Scripts:** `analyze_eval_run.py`  
**Notebook:** `nbs/analyze_eval_run.py`  
**Modules:** `dr_code.analysis.*`

---

## Pipeline (dr-queues)

Parse → test workflow with Mongo-backed lifecycle state and derived file
exports under `exports/runs/{run_id}/`.

| Component | Location |
|-----------|----------|
| Workflow definition | `dr_code.pipeline.definition` |
| Handlers | `dr_code.pipeline.handlers` |
| Seeding | `dr_code.pipeline.seed` |
| Lifecycle orchestration | `dr_code.pipeline.lifecycle` |
| Worker tuning | `dr_code.pipeline.tune`, `scripts/tune_test_workers.py` |

**Mongo collections:**

- `run_manifests` — dr-queues run manifests
- `pipeline_events` — dr-queues lifecycle telemetry
- `eval_results` — upserted `TestOutcome` keyed by `(run_id, sample_id)`

**Scripts:** `demo_pipeline.py`, `eval_run.py`, `tune_test_workers.py`

See [Pipeline runbook](./pipeline-runbook.md) for commands, proof acceptance, and tuning results.

Manual smoke verification on 2026-06-22 covered CLI help, preflight,
in-process runs, split detached lifecycle commands, one-shot detached runs,
export reconstruction, worker cleanup, dump-backed seeding, and stage 4 analysis
joins. The log is in `.scratch/eval-run-lifecycle/manual-testing-2026-06-22.md`.

---

## Major design decisions

1. **Unified `AttemptRecord`** for pool and fresh sources — slice on `provenance.source`, never merge pass rates blindly.
2. **Lightweight HumanEval+ loader in dr-code** — snapshot-first, offline-capable.
3. **code-eval `EXTRACTION_CONFIG`** — not `DEFAULT_CONFIG` at pool scale; use `best_valid_source()` for selection.
4. **dr-providers for fresh generation only** — DSPy deferred.
5. **Stub-as-description for v1 fresh runs** — official prompt stub, not encoder output.
6. **dr-queues + Mongo from v1** — not a throwaway prototype.
7. **Local fork execution** — reset candidate state by child process exit, not Docker container teardown.
8. **Dedup-aware seeding** — `occurrence_count` preserved for weighted analysis.
9. **Transport-agnostic schemas** — `AttemptRecord`, `ParseOutcome`, `TestOutcome` independent of queue layout.
10. **Analysis offline** — export-first; repeatable without live infrastructure.

---

## code-eval integration

Frozen at **`v0.1.1`** / tag **`v0.1.1-frozen`** on sibling `../code-eval`. Path dep in `pyproject.toml` (PyPI deferred — name conflict).

| Requirement | How dr-code uses it |
|-------------|---------------------|
| Pool-scale parse | `EXTRACTION_CONFIG` (`normalizers=()`) |
| Best extracted code | `ValidationResult.best_valid_source()` |
| Provenance | Project from `best_valid_candidate()` into `ParseOutcome.code_eval` |
| Slim storage | Store `ParseOutcome`, not full `ValidationResult` |

**Ruff override:** dr-code uses `ruff>=0.15.18` via `[tool.uv] override-dependencies`; safe because `EXTRACTION_CONFIG` skips normalization subprocesses.

Extend code-eval upstream for behavior gaps; do not fork parse logic into dr-code. More detail: [code-eval investigation](../investigation/code-eval.md).

---

## Local dependencies

| Package | Source | Used in |
|---------|--------|---------|
| code-eval | `../code-eval` editable, `0.1.1` | Stage 2 |
| dr-providers | `../dr-providers` editable, `0.1.0` | Stage 1b |
| dr-queues | `../dr-queues` editable | Pipeline |
| zstandard | PyPI | Stage 4 |

---

## Repository layout

```text
src/dr_code/
  datasets/          # HumanEval+ loader, pool loader, export
  generation/        # dr-providers batch runner
  models/            # AttemptRecord, ParseOutcome, TestOutcome
  parsing/           # code-eval adapter
  testing/           # HumanEval test parser and local fork runner
  pipeline/          # dr-queues workflow, lifecycle, handlers, tune, export, report
  analysis/          # zstd joins, aggregates
scripts/             # typer CLIs per stage + pipeline + tune
configs/             # openrouter_profiles.yaml
nbs/                 # marimo analysis notebook
docs/
  investigation/     # sibling repo notes
  plans/             # this directory
exports/runs/        # derived run artifacts (gitignored)
```

---

## Proof bar results (HumanEval/0–4)

Run `proof-20840125` on 20260621_manual pool dump:

| Metric | Value |
|--------|-------|
| Jobs | 5,828 / 5,828 |
| Wall time | ~16.6 min |
| Tested pass rate | ~25.3% (pool mess expected) |
| Join failures | 0 |
| Outcomes | 5,810 tested, 13 skipped, 5 infra_error |

**Tuning (Mac Mini):** optimal `test=8` workers (~6.6 samples/sec). Recommended production flags: `--workers parse=8,test=8`. Details in [runbook tuning section](./pipeline-runbook.md#live-test-worker-tuning).

---

## Future steps (out of initial scope)

### Full pool replay

Remaining ~163 HumanEval task indices (~172k dedup unique strings). Use `scripts/eval_run.py run`, expand `--task-indices`, or add `--all-tasks`.

### dr-bottleneck integration

Replace AST-only evaluate with dr-code stages 2–4 (or shared libraries). dr-bottleneck keeps enc/dec orchestration; dr-code owns eval semantics.

### DSPy encoder optimization

```text
encoder prompt (DSPy) → encode → decode → stages 2–4 → f(zstd22, pass_rate, …)
```

Prerequisites: `fresh_encoded` generation mode, train/dev/eval splits.

---

## Resolved decisions

| Question | Resolution |
|----------|------------|
| Mongo layout | Both `pipeline_events` and `eval_results` |
| dr-queues dep | Editable path on `../dr-queues` |
| Test parallelism | One child process per sample; tune `test` worker count empirically |
| Run manifest | Mongo `run_manifests` via dr-queues |
| Parse failures | Forward to test with skip; handler catches code-eval exceptions |
| Analysis input | Export-first Parquet/JSONL; Mongo query optional later |
