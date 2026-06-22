# Evaluation run lifecycle refactor

Status: ready-for-agent

## Goal

Deepen the Evaluation run lifecycle module so init, seed, workers, wait, status, export, and one-shot run behavior share one interface and MongoDB remains the source of truth for lifecycle state.

## Decisions

- Migrate dr-code dataclasses to Pydantic first.
- Use frozen Pydantic result models; builders accumulate locally and return final models.
- Keep this refactor inside dr-code.
- Add one lifecycle module at `src/dr_code/pipeline/lifecycle.py`.
- Add one Typer CLI at `scripts/eval_run.py`.
- Delete `scripts/run_eval_pipeline.py` and update docs to the new command.
- MongoDB/dr-queues job state and events are lifecycle state; files under `exports/runs/{run_id}/` are derived artifacts only. See `docs/adr/0001-mongo-backed-evaluation-run-state.md`.

## Issue order

1. Pydantic migration
2. Lifecycle module skeleton
3. Mongo-backed seed/export
4. Worker lifecycle commands
5. One-shot run + docs cleanup
