# Directional HumanEval task difficulty

This workflow estimates which HumanEval tasks appear consistently easy or
hard in the historical generation corpus without evaluating every generation.
Run the numbered scripts in order.

## Corpus input

Stage 1 reads a validated **generation corpus bundle** produced by
[`scripts/build_generation_corpus.py`](../../../build_generation_corpus.py).
See [`docs/generation_corpus.md`](../../../docs/generation_corpus.md) for the
full build command and audited populations.

By default, stage 1 uses the repo fixture at
`tests/fixtures/generation_corpus/human_eval`. Override the bundle with either:

- `DR_CODE_GENERATION_CORPUS_BUNDLE=/path/to/human_eval/bundle`, or
- `--corpus-bundle /path/to/human_eval/bundle` on stage 1.

For baseline runs that will be compared before and after other changes, use
[`run_baseline.sh`](run_baseline.sh) or pin the bundle manifest SHA-256 with
`DR_CODE_EXPECTED_MANIFEST_SHA256` and record the manifest summary emitted in
the stage-1 log.

Example production build:

```bash
DUMP_DIRECTORY=/path/to/20260621_manual
OUTPUT_DIRECTORY=/path/to/corpora/human_eval
TASK_SOURCE=/path/to/humanevalplus_snapshot.json

uv run python scripts/build_generation_corpus.py human_eval \
  --dump-directory "${DUMP_DIRECTORY:?}" \
  --task-source "${TASK_SOURCE:?}" \
  --output-directory "${OUTPUT_DIRECTORY:?}"
```

## Workflow

1. `uv run python scripts/verification/task_difficulty/01_build_eligible_corpus.py`
   loads the pinned generation corpus bundle, keeps the three comparable
   `(generation_mode, budget_mode)` settings, runs exhaustive preprocessing
   across `--workers` worker processes (default 16), and stores generations
   with at least one compilable top-level-function candidate. Workers return
   each output's candidate sources rather than its whole trace, and stage 1
   consumes them as they complete rather than retaining them.
2. `uv run python scripts/verification/task_difficulty/02_select_balanced_sample.py`
   selects one deterministic generation for each available task, setting, and
   model in the fixed model roster.
3. Provision a copied evaluation interpreter and run the evaluator:

   ```shell
   python3 -m venv --copies .evaluation-venv
   uv pip install --python .evaluation-venv/bin/python3 .
   DR_CODE_EVALUATION_PYTHON="$PWD/.evaluation-venv/bin/python3" \
     uv run python scripts/verification/task_difficulty/03_evaluate_sample.py \
       --workers 16 --timeout-seconds 120
   ```

   Stage 3 runs one monolithic [`evaluate_batch`](../../../src/dr_code/evaluation/batch.py)
   over the full selected sample. Candidate parallelism comes from dr-exec's
   `ExecutionPool` scheduling jobs across the whole batch; `--workers` sets the
   global pool capacity and `--timeout-seconds` bounds each candidate execution
   budget.

   Artifacts for a workers/timeout combination live under
   `explicit-runtime/workers-<N>_timeout-<T>/`:

   | Path | Role |
   |---|---|
   | `evaluation_bundles/run/` | Terminal evaluation bundle |
   | `execution_cache.sqlite3` | Persistent execution-cache checkpoints |
   | `evaluation_object_store.sqlite3` | Object store for sample records |
   | `run_manifest.json` | Recovery pins (settings fingerprint, bundle path) |
   | `candidate_results.parquet` | Flat projection export for summarizer |

   **Partial recovery:** if stage 3 stops before publishing a complete bundle,
   re-run the same command without changing workers, timeout, or corpus pins.
   The persistent execution cache skips candidates for samples that already
   finished in a prior attempt.

   Changing workers or timeout starts an independent experiment directory.
4. Run the summarizer with the same flags:

   ```bash
   uv run python scripts/verification/task_difficulty/04_summarize_results.py \
     --workers 16 --timeout-seconds 120
   ```

   It reads `candidate_results.parquet` and reports generation- and task-level
   success rates.

## Execution primitives

The two parallel stages of this workflow run on different dr-exec execution
modes, because they are different kinds of work. dr-exec's
[parallelism guide](https://github.com/danielle-rothermel/dr-exec#choosing-an-execution-mode-a-parallelism-guide)
explains the modes in full.

- **Candidate and test execution (stage 3) — spawned subprocess jobs.** Each
  candidate is model-produced source this workflow did not write, so it runs in
  its own freshly spawned process with its own wall-time budget and its own
  durable execution record. The process boundary is the point: a candidate that
  crashes, hangs, or exits takes nothing else with it, and its outcome is
  evidence attributable to that candidate alone. Startup cost is noise next to
  running a test suite.
- **Preprocessing (stage 1) — dr-exec worker pool.** Preprocessing is
  first-party trusted code that costs single-digit milliseconds per output.
  Spawning an interpreter per output would spend all the cores on `import`, and
  running it on threads in one process would serialize under the GIL. The pool
  starts N long-lived worker processes, each importing the preprocessing entry
  point once, then feeds them jobs over pipes: real cores, no per-item startup.
  It makes no containment claim, which is fine because nothing here is
  untrusted. Workers return only the candidate sources stage 1 consumes: the
  parent decodes and validates every returned byte single-threaded, so
  returning whole traces would cost about a hundred times the payload and cap
  throughput at the parent's parse rate regardless of worker count.

All inputs, outputs, and sampling choices are fixed in `workflow_settings.py`.
Heavy run artifacts and evaluation caches are written under
`~/drotherm/data/.codex/dr-code/task-difficulty-directional/runs/<baseline-name>/`
(override with `DR_CODE_TASK_DIFFICULTY_RUN_DIR`). Git-tracked baseline run
config and results live under
[`baseline/`](baseline/README.md).

## Baseline runner

Run the full workflow and export git-tracked run config and results in one command:

```bash
scripts/verification/task_difficulty/run_baseline.sh pre-106
```

See [`baseline/README.md`](baseline/README.md) for the pinned production corpus,
fixture smoke run, and exported artifact layout. Verbose stage logs stay in the
run directory outside the repository.

Test success is conditional on nonblank output and successful preprocessing.
The preprocessing summary preserves the eligibility denominator. Observed
`0/N` and `N/N` task outcomes are directional extremes, not proof that a task
always fails or always succeeds.
