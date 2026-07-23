# Changelog

## 2026-07-23

- Added a re-runnable `dr-code classify-failures` command that labels
  preprocessing parse/extraction failures (and test failures when evaluation
  artifacts exist) with a versioned, seeded taxonomy using a subscription LLM
  lane (pi headless; glm-coding by default, kimi/minimax/stepfun selectable).
  Each item is classified over N repeats with majority vote, ties resolving to
  `other`, and per-item agreement recorded as a descriptive statistic; lane and
  off-taxonomy errors become recorded typed failures rather than fabricated
  labels. Per-task rollups persist through the machine task-annotation path
  (`origin=machine`, provenance carrying model/taxonomy_version/repeats/mean
  agreement plus per-label counts and the detail-artifact path); per-example
  detail is written to a deterministic JSONL beside the viewer database. The
  feature is additive: no schema migration and no new example-level columns.
  Per-task rollups never overwrite an existing human task annotation (the
  machine rollup is skipped when a human row is present, and the skipped-human
  collision count is surfaced on the run summary); machine-over-machine
  overwrite stays allowed so re-runs refresh their own rows. A tie for the
  dominant per-task label resolves to the rollup-only `mixed` category
  (deliberately not a taxonomy label), with the per-label tie recorded in
  `provenance.extra`.

## 2026-07-22

- Replaced per-candidate OCI container execution with a fresh bounded host
  subprocess using isolated Python mode, a minimal child environment, and
  process-group cleanup on timeout or output overflow.
- Updated production candidate evaluation to preflight NumPy and canonical
  HumanEval+ solutions locally and to record
  `subprocess:python-isolated@v1`; the legacy `sandbox_image` manifest field is
  retained as `null`, and older OCI execution state requires a new run.
- Removed container image preparation from CI and the obsolete HumanEval+
  image build, and documented that host subprocesses must run on disposable,
  constrained workers because they are not security sandboxes.

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
