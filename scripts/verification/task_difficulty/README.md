# Directional HumanEval task difficulty

This workflow estimates which HumanEval tasks appear consistently easy or
hard in the historical generation corpus without evaluating every generation.
Run the numbered scripts in order.

1. `uv run python scripts/verification/task_difficulty/01_build_eligible_corpus.py`
   eagerly loads the historical corpus, keeps the comparable legacy generation
   cohort, runs exhaustive preprocessing, and stores generations with at least
   one compilable top-level-function candidate.
2. `uv run python scripts/verification/task_difficulty/02_select_balanced_sample.py`
   selects one deterministic generation for each available task, setting, and
   model in the fixed model roster.
3. On a disposable worker only, provision a copied evaluation interpreter and
   run the evaluator:

   ```shell
   python3 -m venv --copies .evaluation-venv
   uv pip install --python .evaluation-venv/bin/python3 .
   DR_CODE_DISPOSABLE_WORKER=1 \
     DR_CODE_EVALUATION_PYTHON="$PWD/.evaluation-venv/bin/python3" \
     uv run python scripts/verification/task_difficulty/03_evaluate_sample.py
   ```

   The evaluator verifies the selected interpreter and a NumPy-dependent
   ground-truth HumanEval case before accepting checkpoints or executing the
   sample. It then uses 16 concurrent task workers. Historical model output
   executes with the worker's permissions. Do not run this stage on a
   workstation containing credentials or valuable data.
4. `uv run python scripts/verification/task_difficulty/04_summarize_results.py`
   combines completed task parts and reports generation- and task-level
   success rates.

All inputs, outputs, and sampling choices are fixed in `workflow_settings.py`.
Artifacts and per-stage logs are written under
`~/drotherm/data/.codex/dr-code/2026-08-06/task-difficulty-directional/`.
Evaluation artifacts use the fresh `explicit-runtime/` subdirectory so prior
results remain intact and cannot be mistaken for results from the configured
runtime.

Test success is conditional on nonblank output and successful preprocessing.
The preprocessing summary preserves the eligibility denominator. Observed
`0/N` and `N/N` task outcomes are directional extremes, not proof that a task
always fails or always succeeds.
