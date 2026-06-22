# One-shot run and docs cleanup

Status: ready-for-agent

## Parent

.scratch/eval-run-lifecycle/PRD.md

## What to build

Finish the Evaluation run lifecycle refactor by adding a composed `run` convenience command, updating the pipeline demo to use the lifecycle module, deleting the old one-shot script, and updating docs to the new command vocabulary.

## Acceptance criteria

- [ ] `scripts/eval_run.py run` supports both in-process and detached modes.
- [ ] `run` composes preflight, init, seed, worker execution, wait, export, and proof reporting through the lifecycle module.
- [ ] Default worker spec is `parse=8,test=8`.
- [ ] Detached `run` starts test workers before parse workers.
- [ ] `scripts/demo_pipeline.py` uses the lifecycle module instead of the old runner module.
- [ ] `scripts/run_eval_pipeline.py` is deleted.
- [ ] README/runbook docs use `scripts/eval_run.py` commands.
- [ ] Old `src/dr_code/pipeline/runner.py` interface is deleted or callers are updated to the lifecycle module.
- [ ] Existing checks pass.

## Blocked by

- .scratch/eval-run-lifecycle/issues/03-mongo-backed-seed-export.md
- .scratch/eval-run-lifecycle/issues/04-worker-lifecycle-commands.md
