# Task-difficulty baselines

Git-tracked logs and summaries for directional HumanEval baseline runs. Heavy
parquet outputs and preprocessing caches stay outside the repository under
`~/drotherm/data/.codex/dr-code/task-difficulty-directional/runs/`.

## One-command baseline

From the repository root, on a **disposable worker**:

```bash
DR_CODE_DISPOSABLE_WORKER=1 \
  scripts/verification/task_difficulty/run_baseline.sh pre-106
```

This runs stages 01–04 against the reviewed production HumanEval bundle, then
exports logs and summaries into `baseline/pre-106/`.

## Pinned production corpus

| Field | Value |
|---|---|
| Bundle | `~/drotherm/data/code-comp/generation-corpora/2026-08-08-reviewed/human_eval` |
| Manifest SHA-256 | `fc6a3e6bb33d446d3ccf98bd33a44b7c05225a79340c643859b0d68f9d3d0728` |
| Generations | 203,407 |

Override the bundle or manifest pin with `DR_CODE_GENERATION_CORPUS_BUNDLE` and
`DR_CODE_EXPECTED_MANIFEST_SHA256`.

## Exported artifacts

Each baseline directory contains:

- `corpus_identity.json` — pinned bundle path and manifest hash
- `summary.json` — structured stage status and headline metrics
- `summary.txt` — human-readable summary
- `logs/` — stage logs plus the combined `run.log`

## Fixture smoke run

```bash
DR_CODE_GENERATION_CORPUS_BUNDLE=tests/fixtures/generation_corpus/human_eval \
DR_CODE_TASK_DIFFICULTY_RUN_DIR=/tmp/task-difficulty-fixture \
DR_CODE_DISPOSABLE_WORKER=1 \
scripts/verification/task_difficulty/run_baseline.sh fixture-smoke
```
