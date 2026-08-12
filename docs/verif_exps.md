# Preprocessing Verification Experiments

## Goal

Estimate how often the current preprocessing pipeline succeeds on real
HumanEval generations before committing resources to full experiments.

## Scope

- Exercise preprocessing only; do not run candidate testing or HumanEval evaluation.
- Analyze one HumanEval task or a deterministic random sample of tasks from
  `generation-corpus.parquet`.
- Exclude null and whitespace-only `decoder_output` values.
- Run exhaustive preprocessing directly, without a trace cache or candidate
  evaluation.

## Preprocessing validation

- Report each task's nonblank-row denominator, successful rows, and
  failure-code counts.
- For a task sample, report the row-weighted aggregate success rate with a
  deterministic percentile bootstrap confidence interval that resamples
  tasks.

## Success criterion

The current preprocessing path can be run and measured on a reproducible task
sample with explicit success denominators and uncertainty estimates.

Single-task results remain a workflow smoke test. Sampled estimates describe
the selected corpus and task-sampling procedure; they do not prove universal
preprocessing success.
