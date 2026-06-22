# Mongo-backed seed and export

Status: ready-for-agent

## Parent

.scratch/eval-run-lifecycle/PRD.md

## What to build

Make seed and export use Mongo-backed Evaluation run lifecycle state. Seed should publish Decoder attempts through dr-queues job state; export should reconstruct Decoder attempts from first-stage job payloads, Parse outcomes from parse stage events, and Test outcomes from terminal job payloads.

## Acceptance criteria

- [ ] `seed_eval_run` accepts both an AttemptRecord export and pool replay inputs.
- [ ] `seed_eval_run` requires an existing Evaluation run manifest and fails clearly when init has not run.
- [ ] Export writes derived artifacts under `exports/runs/{run_id}/` without treating those files as lifecycle state.
- [ ] Partial export after parse completion can write attempts and Parse outcomes before terminal completion.
- [ ] Final export writes Test outcomes and a proof report after terminal completion.
- [ ] Proof report wall time uses first stage-start event to last terminal event.
- [ ] Existing checks pass.

## Blocked by

- .scratch/eval-run-lifecycle/issues/02-lifecycle-module-skeleton.md
