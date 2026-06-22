# dr-code plans

Design docs for the compression–correctness evaluation pipeline.

## Read first

1. [Investigation synthesis](../investigation/synthesis.md) — starting state of sibling repos (June 2026).
2. [Overview](./overview.md) — goals, decisions, stage map, future work.
3. [code-eval integration status](../code-eval-work-needed.md) — prelim freeze checklist and dr-code path dep (when working on stage 2).

## Stage design docs

| Stage | Doc | Status | Summary |
|-------|-----|--------|---------|
| 1 | [Generation & dataset](./stage-01-generation-dataset.md) | **Done** (2026-06-21) | Unified raw-generation dataset from pool replay or fresh dr-providers runs |
| 2 | [Parsing](./stage-02-parsing.md) | **Done** (2026-06-21) | code-eval adapter, unit tests, parse CLI, demo (`scripts/demo_stage2.py`) |
| 3 | [Testing](./stage-03-testing.md) | **Done** (2026-06-21) | nl-code adapter, `TestOutcome`, unit tests, `scripts/test_attempts.py`, `scripts/demo_stage3.py` |
| 4 | [Analysis](./stage-04-analysis.md) | Planned | zstd compression joins, aggregates, marimo exploration |

Implement stages in order. Stages 2–3 share the [dr-queues](https://github.com/danielle-rothermel/dr-queues) pipeline runtime from the initial version (not a later add-on).

**Picking up Stage 4?** Start with [Stage 4 handoff](./stage-04-handoff.md).

## Future (out of initial scope)

Described in [Overview → Future steps](./overview.md#future-steps-out-of-initial-scope):

- dr-bottleneck integration (reuse stages 1–4 at scale)
- DSPy encoder optimization (compression + correctness objective)
