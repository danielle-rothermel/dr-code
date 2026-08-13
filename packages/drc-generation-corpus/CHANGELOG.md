# Changelog

All notable changes to `drc-generation-corpus` are documented here.

## 0.2.0 - 2026-08-13

Initial standalone release extracted from the dr-code 0.1.x monolith.

### Breaking changes (from dr-code monolith)

- Import path: `dr_code.generation_corpus` → `drc_generation_corpus`.
- HumanEval task material resolves through `drc_humaneval` instead of
  `dr_code.humaneval`.

### Notes

- Depends on `dr-code` 0.2.0 and `drc-humaneval` 0.2.0.
- Not published to PyPI in this release; install from the workspace or VCS.
