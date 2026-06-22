# dr-code plans

Design and operations docs for the compression–correctness evaluation pipeline.

## Read first

1. [Investigation synthesis](../investigation/synthesis.md) — starting state of sibling repos (June 2026).
2. [Overview](./overview.md) — architecture, stages, design decisions, proof results.
3. [Pipeline runbook](./pipeline-runbook.md) — commands, tuning, troubleshooting.

## Investigation notes (sibling repos)

| Doc | Topic |
|-----|-------|
| [code-eval](../investigation/code-eval.md) | Parse library internals |
| [nl-code](../investigation/nl-code.md) | Docker test execution |
| [dr-providers](../investigation/dr-providers.md) | Fresh generation transport |
| [dr-llm pool](../investigation/dr-llm-humaneval-pool.md) | Pool dump artifacts |
| [dr-bottleneck](../investigation/dr-bottleneck.md) | Related enc/dec experiments |

## Status

All initial-scope work is complete: stages 1–4, dr-queues pipeline, proof bar on HumanEval/0–4. Next operational step: full pool replay at tuned worker counts (see runbook).
