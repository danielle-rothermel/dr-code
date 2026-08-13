# Changelog

All notable changes to the `dr-code` core wheel are documented here.

## 0.2.1 - 2026-08-13

### Fixed

- Restore PyPI package metadata by adding `packages/dr-code/README.md`, so
  release `twine check --strict` passes again after the workspace split.

## 0.2.0 - 2026-08-13

### Breaking changes

- Split the monolithic wheel into a uv workspace. Core `dr-code` now ships only
  `dr_code` (preprocessing, trace, metrics, caching, evaluation machinery).
  Domain packages are separate workspace members:
  `drc-humaneval`, `drc-synthetic`, and `drc-generation-corpus`.
- Removed `dr_code.humaneval`, `dr_code.synthetic`, and
  `dr_code.generation_corpus`. Install the matching `drc-*` wheel or workspace
  member for those capabilities.
- Core evaluation uses generic candidate job contracts
  (`CandidateJobRequest`, `CandidateJobResult`, `CandidateEvaluatorSuite`).
  HumanEval-specific types and the `code_test` metric operator moved to
  `drc-humaneval` and register through entry points
  (`dr_code.metric_operators`, `dr_code.candidate_job_builders`).
- `EvalBatchRequest.preprocess_mode` is required and chooses how sample inputs
  are prepared: `process_pool` runs distinct texts through `preprocess_batch`;
  `in_process` prepares each sample on the caller thread via the bound runner.
- Restructured evaluation batch inputs into nested `SlotData` with
  discriminated `SampleData` or `SampleWithCandidatesData` payloads, replacing
  the flat `SampleEvalInput` / `FrozenCandidateEvalInput` union. Persisted
  request JSON now nests payload fields under `data`; inner `kind` values
  remain `"sample"` and `"frozen_candidates"`.

The monorepo changelog at the repository root retains the full project history.
