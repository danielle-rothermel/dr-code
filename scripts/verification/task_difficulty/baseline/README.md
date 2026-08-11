# Task-difficulty baselines

Git-tracked **run config** and **results** for directional HumanEval baseline
runs. Heavy evaluation bundles, preprocessing caches, execution caches, and verbose
stage logs stay outside the repository under
`~/drotherm/data/.codex/dr-code/task-difficulty-directional/runs/`.

Stage 3 publishes one terminal evaluation bundle under
`explicit-runtime/workers-<N>_timeout-<T>/evaluation_bundles/run/` and exports
`candidate_results.parquet` for the summarizer. Re-run stage 3 with the same
workers, timeout, and corpus pins to resume after an interrupted evaluation.

## One-command baseline

From the repository root:

```bash
scripts/verification/task_difficulty/run_baseline.sh pre-106
```

This runs stages 01–04 against the reviewed production HumanEval bundle, then
exports into `baseline/pre-106/`:

| File | Contents |
|---|---|
| `run_config.json` | Pinned corpus, manifest hash, git rev, workers/timeout |
| `results.json` | Stage completion and headline preprocessing/sampling/eval metrics |
| `results.txt` | Human-readable summary of `results.json` |

Stage logs are written only under the run directory (not git-tracked).

## Pinned production corpus

| Field | Value |
|---|---|
| Bundle | `~/drotherm/data/code-comp/generation-corpora/2026-08-08-reviewed/human_eval` |
| Manifest SHA-256 | `fc6a3e6bb33d446d3ccf98bd33a44b7c05225a79340c643859b0d68f9d3d0728` |
| Generations | 203,407 |

Override the bundle or manifest pin with `DR_CODE_GENERATION_CORPUS_BUNDLE` and
`DR_CODE_EXPECTED_MANIFEST_SHA256`.

## Fixture smoke run

```bash
DR_CODE_GENERATION_CORPUS_BUNDLE=tests/fixtures/generation_corpus/human_eval \
DR_CODE_TASK_DIFFICULTY_RUN_DIR=/tmp/task-difficulty-fixture \
scripts/verification/task_difficulty/run_baseline.sh fixture-smoke
```
