# Preprocessing Verification Experiments

## Goal

Run a fast, bounded characterization of the current preprocessing pipeline on real HumanEval generations before committing resources to full experiments.

## Scope

- Exercise preprocessing only; do not run candidate testing or HumanEval evaluation.
- Select one HumanEval task ID from `generation-corpus.parquet`.
- Exclude null and whitespace-only `decoder_output` values.
- Preserve each remaining generation's `sample_id` and metadata for inspection and comparison.

## Preprocessing validation

- Confirm the pipeline produces the expected traces, extracted candidates, diagnostics, and failure classifications.
- Inspect malformed outputs, unexpected classifications, missing provenance, and awkward operational behavior.

## Cache validation

- Run the exact input slice once with a cold cache and again with a warm cache.
- Confirm the warm run uses cached preprocessing results and produces identical outputs.
- Measure elapsed-time savings and avoided preprocessing work; timing alone is not evidence of a cache hit.

## Success criterion

The current preprocessing path can be run, inspected, repeated, and measured on real data with enough confidence to identify necessary changes before broader experiments.

This single-task run is a workflow smoke test, not evidence that preprocessing handles the full corpus correctly.
