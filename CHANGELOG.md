# Changelog

## 2026-08-05

- Trace persistence requires schema version 3.
- Trace construction snapshots the supplied values mapping and deep-copies
  step facts, so later caller mutation cannot change an existing trace.
  `JsonArtifact.payload` is held by reference and is documented as outside
  that snapshot.
- Step facts widened from string values to validated finite JSON: string
  keys, finite floats, no container cycles, and no non-JSON values, enforced
  at both trace construction and the persistence boundary.
- `Absent` carries a required `failure_code`, a producer-owned string naming
  the failure kind; preprocessing steps raise `StepFailedError` with an
  explicit code and the runner propagates it unchanged.
- Code candidates are nested records — `ExtractionOperation`,
  `CandidateOrigin`, and `CodeCandidate` — carrying an ordered lineage that
  extraction and elementwise steps extend as they transform a source.
- Added the inspected-candidate artifact: `CandidateInspection`,
  `InspectedCodeCandidate`, and `InspectedCodeCandidateSetArtifact`, whose
  inspection fields are structural only.

## 2026-08-04

- Removed dead public API and unreachable code paths across the HumanEval,
  preprocessing, synthetic, and trace packages, including their re-exports and
  placeholder residue.
- Corrected docstrings and comments that described behavior the code no longer
  has, so every module, step, and operator description matches what it does.
- The HumanEval identifier spelling is uniform across modules, names, and
  tests.
- The shared model base module is `dr_code.base` and the schema command-line
  entry point is `dr_code.schema_cli`.
- Misplaced modules and tests live with the code they belong to: operator
  settings sit with the operator base, the synthetic corruption registry has
  its own module, and the metrics policy example and trace import probe live in
  the test suite instead of the wheel.
- Tests mirror the source layout: the flat `tests/unit` directory is gone, the
  metrics registry has its own module, and preprocessing tests import each
  function by its own name from its owning module.
- Snapshot loading takes an explicit snapshot path argument instead of
  resolving one implicitly.

## 2026-08-03

- Established explicit manual component coordinates and reset every current
  production preprocessing step, metric operator, preprocessing definition,
  parser profile, scoring profile, and metrics profile to version `"0"`.
- Added a tooling-only component-version development marker and an exhaustive
  repository contract test that keeps registered production components at the
  initial version while the marker is enabled.
- Removed public semantic hashing, definition hashes, hash-bearing trace and
  metric-record fields, and the public execution-request cache digest. Cache
  reuse is now private mechanics tested through execution behavior.
- Structured preprocessing producers now include the complete ordered step
  composition, and metric records nest both that producer coordinate and the
  complete ordered metrics definition. Trace persistence requires schema
  version 2.
- HumanEval snapshots carry the explicit versioned override set, and synthetic
  samples carry independently versioned recipes and ordered corruption
  coordinates.
- Strengthened the CI and deterministic test baseline with locked dependency
  sync, formatting and type checks, current action versions, isolated module
  execution, state-based timeout translation, and functional boundary tests.

## 2026-07-15

- Removed superseded design-review and eval-flow hero artifacts, and updated
  the preprocessing and metrics plans to reflect the current HumanEval
  boundaries and consumers.
- Updated the package description to cover producer-blind HumanEval+
  execution and synthetic corruption datasets.

## 2026-07-14

- Split HumanEval test parsing and sandbox batch orchestration out of
  `task.py`, extracted the sandbox runner into a dependency-free Python
  resource, and injected `SandboxRunner` through the evaluation entry points.
- Removed the dead `serve`/explain facade and replaced the repo-specific
  viewer surface with domain-agnostic viewer primitives and a gallery.
- Made escaped-newline tests assert behavior instead of trace-node names,
  documented the extraction-trace contract, made corruption implementations
  explicit ABCs, and warmed sandbox containers before timed probes.
- Added the eval-flow design and implementation plans and the serializable
  `dr_code.trace` boundary contract package.

## 2026-07-13

- Moved HumanEval candidate execution into a fail-closed OCI sandbox with no
  host mounts or network, an unprivileged read-only filesystem, bounded JSON
  IPC and resources, and complete container cleanup on timeout.
- Hardened sandbox scoring attribution: resource-limit kills, `SystemExit`,
  and output floods now score as candidate errors instead of harness
  failures, runner case ids must be known and unique (duplicate rows can no
  longer inflate coverage), runner output fields are clipped to fit the IPC
  bound, and the real sandbox probes always run when `CI` is set.

## 2026-07-09

- Completed the transform/analysis library grid: `dr_code.code_analysis` and
  `dr_code.text_analysis` join `code_transforms`/`text_transforms`, with
  enumeration (annotation/docstring/signature sites, function locals, block
  segmentation) exported separately from the transforms that apply policy.
- Removed `canonicalize` (alias of `strip_docstrings`) and the
  `NodeTransformer` implementation classes; `parsed_code` boundary models
  became declarative with explicit constructors.
- Renamed corruption vocabulary to `dr_code.synthetic.corruptions`
  (`Corruption`, `CorruptionName`, `Recipe.corruptions`).

## 2026-07-08

- Cut over the repo to the producer-blind evaluator shape: the batch pipeline,
  `code_eval` dependency path, flatten path, golden/corpus behavior pins, and
  DSPy handling were removed from the active tree.
- Kept the synthetic generator as a first-class CLI for deterministic
  corruption datasets.
- Standardized forward-facing vocabulary on submission instead of generation
  for evaluator inputs and docs.
