# Lifecycle module skeleton

Status: ready-for-agent

## Parent

.scratch/eval-run-lifecycle/PRD.md

## What to build

Create the deep Evaluation run lifecycle module interface without changing the current run behavior yet. The module should expose Pydantic result models and functions for init, status, preflight, and the named lifecycle operations so future slices have one seam to deepen.

## Acceptance criteria

- [ ] `src/dr_code/pipeline/lifecycle.py` exposes the agreed lifecycle interface: `init_eval_run`, `seed_eval_run`, `start_eval_workers`, `stop_eval_workers`, `wait_for_eval_run`, `get_eval_status`, `export_eval_run`, and `run_eval_once`.
- [ ] New lifecycle result objects are frozen Pydantic models.
- [ ] Initial functions delegate to existing dr-code/dr-queues behavior where possible instead of duplicating implementation.
- [ ] Existing checks pass.

## Blocked by

- .scratch/eval-run-lifecycle/issues/01-pydantic-migration.md
