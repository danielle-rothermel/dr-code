# Changelog

All notable changes to `drc-synthetic` are documented here.

## 0.2.0 - 2026-08-13

Initial standalone release extracted from the dr-code 0.1.x monolith.

### Breaking changes (from dr-code monolith)

- Import path: `dr_code.synthetic` → `drc_synthetic`.
- Console script: `dr-code-synthetic` → `drc-synthetic`.
- `SyntheticSampleCoordinate`, `RecipeCoordinate`, and `CorruptionCoordinate`
  are core-owned provenance types in `dr_code.evaluation.provenance`.

### Notes

- Depends on `dr-code` 0.2.0 and `drc-humaneval` 0.2.0.
- Not published to PyPI in this release; install from the workspace or VCS.
