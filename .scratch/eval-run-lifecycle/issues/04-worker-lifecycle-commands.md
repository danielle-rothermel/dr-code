# Worker lifecycle commands

Status: ready-for-agent

## Parent

.scratch/eval-run-lifecycle/PRD.md

## What to build

Add `scripts/eval_run.py` lifecycle commands for detached worker operation: preflight, init, seed, start, stop, wait, status, and export. The CLI should be a thin adapter over the lifecycle module; split lifecycle worker start is detached-only, and wait owns the terminal tap.

## Acceptance criteria

- [ ] `scripts/eval_run.py` has subcommands for `preflight`, `init`, `seed`, `start`, `stop`, `wait`, `status`, and `export`.
- [ ] `start` starts detached stage workers and can start parse or test independently.
- [ ] `stop` explicitly stops selected workers or stages; `wait` has no hidden stop side effects.
- [ ] `wait --target terminal` records terminal events through the terminal tap.
- [ ] `status` prints a compact eval-specific summary and supports JSON output.
- [ ] The CLI uses lifecycle functions rather than calling dr-queues directly except where the lifecycle module explicitly owns the adapter.
- [ ] Existing checks pass.

## Blocked by

- .scratch/eval-run-lifecycle/issues/02-lifecycle-module-skeleton.md
- .scratch/eval-run-lifecycle/issues/03-mongo-backed-seed-export.md
