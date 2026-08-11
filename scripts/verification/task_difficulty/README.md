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
   `(generation_mode, budget_mode)` settings, runs exhaustive preprocessing,
   and stores generations with at least one compilable top-level-function
   candidate.
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

   The evaluator verifies the selected interpreter and a NumPy-dependent
   ground-truth HumanEval case before accepting checkpoints or executing the
   sample. `--workers` controls concurrent task workers and
   `--timeout-seconds` bounds each candidate execution budget. The final log
   line reports both end-to-end and evaluation-only seconds per selected
   generation.
   Before spawning workers, the evaluator raises its own soft open-file limit
   to a worker-scaled minimum when the process hard limit permits it. If that
   is impossible, it exits before evaluation with the required `ulimit`
   command.
   Each workers/timeout combination writes to its own experiment directory,
   so changing either flag starts an independent run without deleting prior
   results.
4. Run the summarizer with the same flags:

   ```bash
   uv run python scripts/verification/task_difficulty/04_summarize_results.py \
     --workers 16 --timeout-seconds 120
   ```

   It combines the matching completed task parts and reports generation- and
   task-level success rates.

All inputs, outputs, and sampling choices are fixed in `workflow_settings.py`.
Heavy run artifacts and preprocessing caches are written under
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
