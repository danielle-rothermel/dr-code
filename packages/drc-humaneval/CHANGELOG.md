# Changelog

All notable changes to `drc-humaneval` are documented here.

## 0.2.0 - 2026-08-13

Initial standalone release extracted from the dr-code 0.1.x monolith.

### Breaking changes (from dr-code monolith)

- Import path: `dr_code.humaneval` → `drc_humaneval`.
- Console script: `dr-code-humaneval-schema` → `drc-humaneval-schema`.
- The `code_test` metric operator registers through the
  `dr_code.metric_operators` entry point instead of the core registry.
- HumanEval candidate job request/result types and the job harness entry point
  live in `drc_humaneval.job` and register through
  `dr_code.candidate_job_builders`.

### Notes

- Depends on `dr-code` 0.2.0.
- Not published to PyPI in this release; install from the workspace or VCS.
