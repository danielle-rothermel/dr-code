# Changelog

## 2026-08-04

- The evaluation kernel names every artifact by manual component coordinates:
  definition references, config references, question identity, task identity,
  and repeat identity are coordinate models compared by plain equality.
  Cryptographic digests appear only in the private execution cache.
- Definition settings and variable values carry a hashable normalized JSON
  value. Objects are key-order independent — entries are stored in name order,
  so two objects with the same names and values are the same value however
  they were written. Arrays are ordered, and both objects and arrays compare
  by exact recursive type, so `1`, `1.0`, and `true` stay distinct at every
  depth. Normalized values serialize as the ordinary JSON they represent.
- Reduced operator resolution to registered name plus manual version, and
  exported the persisted metric question, definition, and record boundary
  models from the metrics facade.
- Collapsed the kernel's parallel question, definition, record, and step
  models onto the persisted boundary models. `MetricsDefinition`,
  `MetricQuestion`, `MetricRecord`, `RecordStatus`, and
  `PreprocessingDefinition` each have exactly one implementation; the kernel's
  authoring surface is now `MetricExtractionTemplate`,
  `MetricQuestionTemplate`, `PreprocessingTemplate`, and
  `PreprocessingStepTemplate`, whose `materialize` resolves variable
  references and yields a config nesting the concrete definition that the
  engine and preprocessing runner execute.
- `extract_metrics` and `extract_metrics_batch` answer a `MetricsDefinition`
  against traces and return `MetricRecord`s; an evaluation procedure is
  supplied as an optional binding that contributes the trace-source contract
  and the live operator-resolution check.
- Added `record_facts`, which projects a measured record onto unit-carrying,
  lineage-stamped `MetricFact`s from the operator's `FACT_UNITS` declaration,
  reporting a declared fact with no value as not-applicable with a reason.

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
