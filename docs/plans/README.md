# dr-code plans

Design docs for the compression–correctness evaluation pipeline.

## Read first

1. [Investigation synthesis](../investigation/synthesis.md) — starting state of sibling repos (June 2026).
2. [Overview](./overview.md) — goals, decisions, stage map, future work.
3. [code-eval integration status](../code-eval-work-needed.md) — prelim freeze checklist and dr-code path dep (when working on stage 2).

## Stage design docs

| Stage | Doc | Summary |
|-------|-----|---------|
| 1 | [Generation & dataset](./stage-01-generation-dataset.md) | Unified raw-generation dataset from pool replay or fresh dr-providers runs |
| 2 | [Parsing](./stage-02-parsing.md) | code-eval extraction/recovery; parse-stage queue workers |
| 3 | [Testing](./stage-03-testing.md) | nl-code Docker execution; test-stage queue workers; Mongo telemetry |
| 4 | [Analysis](./stage-04-analysis.md) | zstd compression joins, aggregates, marimo exploration |

Implement stages in order. Stages 2–3 share the [dr-queues](https://github.com/danielle-rothermel/dr-queues) pipeline runtime from the initial version (not a later add-on).

## Future (out of initial scope)

Described in [Overview → Future steps](./overview.md#future-steps-out-of-initial-scope):

- dr-bottleneck integration (reuse stages 1–4 at scale)
- DSPy encoder prompt optimization (compression + correctness objective)
