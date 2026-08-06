# Changelog

## 0.1.6 - 2026-08-06

- The cache benchmark uses the existing SQLite whole-trace preprocessing
  cache for cold and warm measurements and verifies exact equivalence with
  uncached traces. Candidate reuse remains a planning-only measurement that
  does not execute candidate code.
- Verification scripts load selected nonblank HumanEval generations eagerly
  with Polars, benchmark exhaustive preprocessing and cache reuse, and report
  task-level bootstrap confidence intervals for candidate and row reuse.
- HumanEval generation-length analysis produces versioned summary data and
  plots for comparing output distributions.

## 0.1.5 - 2026-08-06

- HumanEval execution uses an explicitly selected Python runtime and declares
  NumPy as a direct dependency of its bundled HumanEval+ test harness.
- The HumanEval runner distinguishes trusted support-code initialization
  failures from candidate setup failures and case outcomes. Metrics report
  harness failures as operator failures while retaining candidate-attributable
  errors as measured evidence.

## 0.1.4 - 2026-08-06

- Execution requests use compact versioned cache keys and batch-prefetch
  planned keys before evaluation. The checkpointed execution cache serves its
  hot path from memory, persists outcomes through dr-store batch operations on
  a background writer, and exposes cache activity statistics.
- HumanEval submission scoring accepts an ordered public batch, plans every
  extracted candidate before one execution-engine call, and keeps the existing
  single-submission scorer as a one-element convenience.
- dr-exec and dr-store are upgraded to 0.1.5 for batch cache persistence.

## 0.1.3 - 2026-08-06

- Publication checks install the built wheel into an isolated environment and
  exercise its public modules, packaged HumanEval driver, version metadata, and
  production-executor construction before upload; supported Darwin runs also
  execute a real `ProcessExecutor` job. Hatchling is pinned exactly, the PyPI
  environment admits release tags only, and the local pre-check refuses a stale
  lockfile without allowing later commands to rewrite it.
- Sandboxed execution is cut over to
  [dr-exec](https://github.com/danielle-rothermel/dr-exec) 0.1.4.
  The execution adapter uses
  [dr-serialize](https://github.com/danielle-rothermel/dr-serialize) 0.1.2
  directly to build the versioned request identity document.
  `dr_code.core.execution.executor` builds
  `ExecutionJob`s (an `UntrustedPythonTarget` driver plus a JSON request
  document) under finite wall-clock, input, and payload-output budgets and a
  fixed environment grant. Exited, signaled, wall-time, and payload-output
  outcomes have fixed mappings; payload-owned protocol failures map to
  candidate-attributable kills; every other outcome fails closed as an
  executor failure. Non-standard JSON constants, JSON numbers that cannot be
  represented finitely, and timeouts too large to represent in nanoseconds
  fail at declaration as `DeclarationError`.
  The injected runner seam is now the dr-exec `Executor` type
  (`executor: Executor | None = None` on `score_humaneval_submission`,
  `evaluate_humaneval_code`, `extract_metrics`, and
  `extract_metrics_batch`), and dr-exec's `ExecutorFailure` joins the metrics
  engine's internal invariant error as a fail-closed infrastructure exception.
  The in-memory per-batch `ExecutionCache` seam is unchanged; dr-exec's
  durable `CachingExecutor` is not wired here.
- The kill classification changed with the transport: the retired container
  exit codes `{137, 139}` are replaced by typed classification. dr-exec
  reports mid-run payload death as a signaled outcome or a payload-owned
  protocol failure, and both surface as `ExecutionKilledError` (a
  candidate-attributable error, sentinel `-100_000_003` in engine execution
  outcomes). Persisted `ExecutionOutcome.returncode` values therefore
  changed meaning, which invalidates any prior execution-cache contents.
- The HumanEval runner script is now a dr-exec driver
  (`runner_driver_script.py` defining `dr_exec_main(request, emit)`); case
  results still travel on redirected stdout, and the protected protocol
  channel carries no outputs yet. An authenticated result channel over the
  protected protocol remains the fix if single-task integrity against
  adversarial candidates becomes a requirement.
- 2026-08-06 — OCI sandbox retirement security ledger. The retired Docker
  sandbox and its test suites provided guarantees that subprocess execution
  does not replace: container image pinning by digest; credential, filesystem,
  network, process, and memory isolation; container lifecycle termination;
  runtime discovery, image inspection, runtime allowlisting, and runtime
  environment construction. Under dr-exec, submitted programs are not
  contained by the subprocess boundary: they retain the invoking worker's
  permissions, external worker isolation is the deployment boundary, and
  evaluations run only on disposable workers. `DR_CODE_SANDBOX_IMAGE`,
  `DR_CODE_RUN_SANDBOX_TESTS`, the `oci` pytest marker, and the dedicated CI
  and release OCI jobs are removed with the sandbox.

## 0.1.2 - 2026-08-06

- Pull-request CI requires every change to advance the project version by one
  patch, keep the root lockfile version synchronized, and carry the matching
  dated changelog entry.
- Tagged releases verify the tagged commit belongs to `main` before validation,
  build, or PyPI publication; protected `v*` tags restrict release-tag creation
  and prohibit tag updates and deletion.
- `dr_code.caching` adds opt-in memoization of preprocessing traces.
  `run_preprocessing_cached` keys raw input text, the bound runner's resolved
  producer coordinate, and the trace schema version through dr-store's
  `derive_cache_key`, stores a serialized trace in a `RecordCache`, and returns
  either a matching hit or the fresh trace. Cache failures, invalid records,
  and entries whose producer or input does not match the request are logged
  and treated as misses.
  No existing call site consults a cache.
- dr-store 0.1.4 supplies the record-cache primitive and the managed
  `SqliteRecordCache` persistent lifecycle.
- The `code_test` operator uses the metrics engine's shared parsed-module view
  during result computation and keeps the validated HumanEval task per exact
  payload for the life of a binding. The recompute-and-reject task validator is
  unchanged.

## 0.1.1 - 2026-08-05

- The package is release-ready at version 0.1.1 with project metadata, a
  source-only sdist, tag-validated PyPI trusted publishing, and Python 3.13 and
  3.14 qualification.
- `.defs/terms.toml` and `.defs/contracts.toml` are the authoritative shared
  vocabulary and standing contracts; the GitHub Pages reference renders both
  files directly and repository tests validate their graphs and public symbol
  mappings.
- Depot CI separates Python, live OCI, and viewer checks, while the installed
  commit hook runs the same non-OCI repository pre-check with pinned uv and
  pnpm versions.
- Source ownership is a hard cut: `dr_code.core.models`,
  `dr_code.core.source`, and `dr_code.core.execution` own the shared model,
  source, and sandbox-execution foundations, with no compatibility aliases.
- Alpha-renaming preserves callable parameter names by default, and top-level
  import removal preserves non-import siblings on semicolon-shared single-line
  statements.
- `dr_code.humaneval` owns its benchmark runner, schema command, and metric
  behavior through `runner`, `schema_cli`, and `metric_operator`.
- `build_dataset` accepts a keyword-only `snapshot_path`; `tasks` and
  `snapshot_path` are mutually exclusive, and omitting both loads the pinned
  HumanEval+ network revision.
- Sandbox execution supports Docker only and requires the digest-pinned image
  locally; Podman is not a supported runtime.
- Trace reads return defensive snapshots, and trace, metric, and evaluation
  boundaries reject malformed namespaces and non-finite persisted values.
- Tests mirror their contract owners: core and HumanEval tests sit with those
  packages, while metrics tests are divided among engine, operator, and record
  contracts and preprocessing step tests sit under `tests/preprocessing/steps`.

## 2026-08-05 (HumanEval evaluation contracts)

- `dr_code.humaneval.batch_runner` owns the HumanEval batch protocol as one
  implementation: `build_humaneval_batch_request` builds every request and
  `interpret_subprocess_batch_result` reads every completed process back into
  case results. The `code_test` metrics operator routes through both and
  assembles an `EvaluationTaskResult`, so request bytes, top-level-function
  selection, best-function selection, and coverage are computed once rather
  than once per scored path. The operator's remaining difference is
  attribution: runner-protocol breakage becomes candidate-attributable case
  errors instead of raising. `tests/metrics/test_operator_parity.py` pins the
  operator's requests byte-identical to the canonical ones.
- The sandbox runner reserves its results channel. It captures the protocol
  stdout handle before candidate code runs, points `sys.stdout` at the bounded
  stderr, and writes results only through `emit_results`, so a candidate that
  prints — including well-formed protocol JSON — does not collide with what
  the host parses; candidate output is preserved on stderr as a diagnostic.
  This contains accidental collisions, not an adversarial candidate, which
  can still forge its own task's case results through the runner's module
  globals or a direct write to file descriptor 1.
- `HumanEvalTask` is a `FrozenModel` carrying `notes` as a tuple. Its
  validator always recomputes `parsed` and `parsed_tests` from
  `prompt + canonical_solution` and `test`, and rejects a supplied value that
  disagrees, so a task can never carry a parse describing other code.
- `EvaluationTaskResult`'s five derived readings (`best_function_name`,
  `failures`, `coverage_complete`, `passed`, `status_counts`) no longer
  serialize, so the model's own dump revalidates under `extra="forbid"` and
  the readings are recomputed from `results`. `EvaluationTaskSummary` remains
  the shape that carries the readings across a boundary.
- `tests/metrics/test_registry.py` derives every operator result model from
  `REGISTRY` — through `compute`'s return annotation and that model's
  subclasses — and asserts `UNITS` matches the field set exactly. This catches
  a unit declared for a field that no longer exists and a field on a variant
  no test exercises, neither of which `to_facts()` can catch at runtime.

## 2026-08-05 (extraction recall)

- `extract_all_representations` reads a JSON `code` field from the response
  as written *and* from the fenced blocks of its answer region, so a
  response that puts its envelope inside a ```` ```json ```` fence still has
  its declared code field read rather than scraped. The answer region is the
  `[[ ## code ## ]]` field when the response marks one and the whole
  response otherwise, so a worked example fenced under a
  `[[ ## prompt ## ]]` is not read as the response's own declaration.
  Decoding stays strict: an envelope is read only when the block is a
  complete JSON object carrying a non-blank string `code`, and nothing is
  repaired.
- `drop_after_last_return` locates the salvage boundary by tokenizing
  rather than by matching lines. A `NEWLINE` token fires only once bracket
  continuations close, so a `return` spanning several lines is kept whole
  instead of being cut mid-bracket, and returns written inside strings or
  comments are not boundaries. It returns `None` when no boundary can be
  located — a malformed token, an unterminated bracket, or text it cannot
  tokenize — so a salvage that cannot be located is never performed.
  `add_last_return_salvage` contributes nothing for such a candidate.
- `infer_missing_imports` inserts inferred imports after the module header
  a candidate already carries: a leading docstring, then contiguous
  `from __future__` imports. An import above either of those is a defect —
  it makes a `from __future__` candidate uncompilable and demotes a module
  docstring to a bare expression.
- The speculative repair parse in import inference suppresses
  `SyntaxWarning` while probing candidate text, so warnings from invalid
  escape sequences in arbitrary decoder output never leak to the caller's
  warning filters. The suppression is scoped to the probe parse only.
- `tests/preprocessing/fixtures/hard_examples.json` pins extraction against
  130 recorded LLM decoder outputs with human verdicts, partitioned into
  development and holdout sets. The cases are evidence, not a
  specification: disagreements are enumerated as individually-reasoned
  strict `xfail`s, so an open decision stays visible and a case that starts
  agreeing fails as `XPASS`.

## 2026-08-05 (evaluation package)

- `dr_code.evaluation` declares and reduces evaluations. It is
  producer-blind and executor-blind: it names no dataset or benchmark, and
  it never resolves anything in a registry.
- Coordinates address one point in an evaluation and nest their parents
  whole, so a persisted artifact is interpretable on its own:
  `DatasetCoordinate`, `TaskSetCoordinate`, `RepeatPlanCoordinate`,
  `SampleCoordinate` (task-set and repeat-plan coordinates plus a task
  identity and repeat index), and `CandidateCoordinate` (a sample, a
  preprocessing-definition coordinate, and a candidate ordinal).
- `CandidateCoordinate.candidate_ordinal` indexes the materialized
  candidate set — after deduplication and after filtering — the same
  definition `MaterializeCandidateSet` establishes.
- A `TaskSet` records the ordered population it selected from alongside the
  selection. Selected identities are unique, drawn from the population, and
  a subsequence of it, so two task sets with identical content compare
  equal.
- A `RepeatPlan` is uniform, contiguous, and task-major: every task gets
  the same number of slots, indices run `0 .. repeats - 1`, and flattening
  runs all repeats of the first selected task before any of the second.
  `seeds` is optional and, when present, carries exactly one seed per slot.
- An `EvaluationProcedure` nests the resolved `PreprocessingDefinition` and
  `MetricsDefinition` rather than their coordinates, because a procedure
  declares work about to run — the deliberate asymmetry with archived
  records, which carry registry-free projections instead.
- `EvaluationPlan` ties a task set, repeat plan, procedure, and
  `AggregationPolicy` together, requiring the repeat plan to cover exactly
  the selected tasks and the aggregated question to be one its own nested
  metrics definition declares.
- `AggregationPolicy` is closed and minimal: the question and fact name
  addressing which number to read, the `AggregationStatistic`, and separate
  `NotApplicablePolicy` rules for not-applicable and operator-failure
  records. No templating and no free-form knobs.
- `aggregate` is pure — no I/O, no registry, no clock, no randomness — and
  takes complete explicit slots, where a slot with no record carries
  `None`. Its five typed results are `AggregationOk`, `AggregationMissing`,
  `AggregationNotApplicable`, `AggregationEmptyDenominator`, and
  `AggregationNonFinite`, discriminated on `status`; none is a sentinel
  float, and overflow is reported as a result rather than raised.
  Discrimination over the record union is a type check.
- `Score` is a derived value that never travels back into a metric fact. It
  carries a finite scalar, a unit from the shared `MetricFactUnit` (never
  `TEXT`, since a score is a measurement), the evaluation it summarizes,
  and the fact coordinates it was computed from.
- Golden tests pin the exact serialized literals of `EvaluationPlan` and
  `Score` as the persisted wire format.

## 2026-08-05 (metric record evolution)

- Metric records carry an explicit `schema_version` of 1, pinned by a
  golden test on the exact serialized literals of a representative record.
- `MetricRecord` is a closed discriminated union keyed on `status`:
  `MeasuredRecord` carries ordered facts, `NotApplicableRecord` nests the
  complete `Absent`, and `OperatorFailureRecord` nests a structured
  `OperatorFailure`. `METRIC_RECORD_ADAPTER` is the loader for persisted
  records.
- Measured answers are ordered `MetricFact` models — name, strict finite
  scalar, and a unit from the closed `MetricFactUnit` enum. Every operator
  result class declares the unit of each field it emits in `UNITS`, and a
  field without a declared unit fails loudly at projection.
- Records nest a shared `MetricRecordIdentity`: the question coordinate,
  the operator version, the trace producer coordinate, and the metrics
  definition coordinate. An identity must name a question its own nested
  definition coordinate declares.
- `MetricQuestionCoordinate` and `MetricsDefinitionCoordinate` are the
  registry-free persisted projections of a declaration. Question settings
  persist as ordered `ComponentSetting` entries — nested settings groups
  named by dotted path — so records validate structurally and archived
  records stay loadable across settings churn and across operator
  implementation and version churn. Metric names stay a closed enum, so the
  guarantee does not extend to records naming a deleted metric.
- `MetricFact` rejects a dot in a fact name, which is what makes the
  two-column `record_rows` fact scheme collision-free.
- `OperatorSettings` lives in `dr_code.metrics.settings`.
- `record_rows` lifts identity fields to top-level columns and emits two
  columns per fact: `"{metric}.{name}"` for the value and
  `"{metric}.{name}.unit"` for its unit.

## 2026-08-05 (preprocessing hard cut)

- Preprocessing binds and runs separately: `bind_preprocessing` validates a
  definition once and returns a `BoundPreprocessingRunner` whose `run`
  folds over any number of inputs. `run_preprocessing` is the one-shot
  wrapper; `bind_external_preprocessing` is the explicit path for
  unregistered definitions.
- The preprocessing facade is curated to the definition models, the
  resolver, the bound runner, the binding functions, and the one-shot
  runners. The step registry, step base classes, and individual step
  mechanics are internal.
- One registered definition, `exhaustive-function-candidates@0`, replaces
  the two previous registered definitions. It reads every supported
  representation additively rather than taking the first that succeeds
  (readings that name a code field explicitly — a JSON `code` key, a
  `[[ ## code ## ]]` marker — contribute before the readings that scrape
  code out of arbitrary text, so a fenced block in another marked field
  cannot shadow the marked answer; the ordering makes no reading
  exclusive), shapes each candidate, adds last-return truncation as an additional
  candidate, repairs and infers imports (after the truncation, so a
  candidate that only becomes parseable once truncated still gets the
  imports its body needs), drops blanks, merges exact duplicates while
  concatenating their lineages, inspects each source exactly once, filters
  on the stored inspection, and materializes every survivor.
- Removed the first-success strategy ladder and its public strategy
  registry, the alternatives step base, destructive last-return
  truncation, single-candidate selection, and the redundant pass-through
  cardinality step.
- Failure codes are the closed `PreprocessingFailureCode` enum, and
  `StepFailedError` carries optional structured JSON evidence that the
  runner records as the failing step's facts.
- `candidate_ordinal` is defined as the index into the materialized
  candidate set, after deduplication and after filtering.
- HumanEval extraction runs the registered definition and applies its own
  acceptance policy, `accept_first_surviving`, over the materialized set.
  The `CodeExtractionResult` model now carries the accepted source, its
  candidate ordinal, the preprocessing trace, and preprocessing's failure
  code. Parser profiles are gone; scoring profiles name a preprocessing
  definition coordinate instead.
- Import repair, inference, and deduplication treat text that cannot be
  encoded — a lone surrogate, a null byte — as unparseable and pass it
  through untouched, so such input is rejected by the compilability filter
  instead of raising out of the pipeline.
- Acceptance semantics diverge from the previous pipeline. Extraction
  requires a top-level function, so lambda-bound and class-wrapped
  solutions are rejected at extraction with
  `no_candidate_survived_filtering` instead of reaching evaluation, and
  scoring's `NO_TOP_LEVEL_FUNCTIONS` outcome is preempted for
  pipeline-extracted candidates.
- Last-return truncation is no longer destructive: the original candidate
  is accepted as written and the truncation is an additional candidate, so
  a solution whose content continues past its last `return` survives.
- Additive extraction recovers responses the first-success ladder missed —
  including unfenced code alongside a fenced snippet, which the
  fenced-or-else-unfenced reading dropped.

## 2026-08-05

- Trace persistence requires schema version 3.
- Trace construction snapshots the supplied values mapping and deep-copies
  step facts, so later caller mutation cannot change an existing trace.
  Artifact payloads, `JsonArtifact.payload` included, are copied at
  validation, so the snapshot covers them too.
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
