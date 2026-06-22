# Mongo-backed Evaluation run lifecycle state

Evaluation run lifecycle state is persisted in MongoDB through dr-queues: manifests, seed batches, job states, worker records, pipeline events, and terminal outcomes. Files under `exports/runs/{run_id}/` are derived artifacts for inspection, analysis, and sharing; they are not required to continue or resume an Evaluation run.

## Considered options

- MongoDB as source of truth, files as exports only — preserves the recent dr-queues move away from file-backed state and keeps continuation state in one place.
- Write run artifacts during seed and use them later as state — simpler export plumbing, but reintroduces file-backed lifecycle assumptions.
