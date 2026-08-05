# Frozen PR audit: #68–#72

Audit of the five frozen draft PRs (`rebuild/05` … `rebuild/09`) against
main after #85–#90. Each PR's increment is its diff against its stacked
parent. The parent stack (PRs 64/66/67) closed unmerged: `dr_code.execution`,
the `dr_code.eval` Definition/Config kernel, pre-v3 trace shapes, and the
flat `MetricRecord` exist only on the frozen branches.

Summary:

| PR | Increment | Executor coupling | Mechanical rebase | Disposition |
|----|-----------|-------------------|-------------------|-------------|
| 68 corpus evaluation | ~11.6k lines | seam-only after rewiring | no — ~10–12% survives | re-derive; gates 69/71/72 |
| 69 preprocessing analysis | ~19.3k lines | none | no | re-derive; gated on 68 |
| 70 behavioral mutants | ~6.6k lines | dead `dr_code.execution` only | no | re-derive core now; no chain deps |
| 71 task annotations | ~8.4k lines | none | no — 0/35 files apply | re-derive after 69 |
| 72 failure classification | ~12.4k lines | none (reads persisted records) | no | taxonomy layer lands now; rest gated on 68/69/71 |

No PR is blocked on dr-exec. The dr-exec interactions are confined to:
executor identity baked into persisted hashes (68, 70), and 72's
`lane.py` provider transport (deferred per F72-1).

## #68 — Durable corpus evaluation

- Executor: imports `dr_code.execution` (`PythonSubprocessRunner`,
  `SubprocessError`, `run_python_subprocess`, raw `.returncode` preflight).
  Main's seam is `SandboxRunner` with `input_json=` (branch: `input_text=`).
  `RUNNER_IDENTITY` bakes executor identity into persisted generation
  hashes.
- Dead dependencies: `dr_code.eval` kernel (`.materialize()`, config
  hashes; `coordinate_validation.py` has no port target), old preprocessing
  (`candidate_id_for_source`, `normalize_decoder_output`, failure codes
  with empty intersection against the current 3-code vocabulary), pre-v3
  trace shapes, flat `MetricRecord`.
- Survives (~10–12%): `atomic_directory.py`, `durability.py`,
  `stable_files.py`, `output_paths.py` (~1,130 lines, dependency-free);
  `CodeTestResult` relational invariants (`Field(ge=0)` +
  `validate_relational_invariants`) as a three-way merge with main's
  `UNITS`.
- Silent-drift constants (all hashed into persisted coordinates):
  `METRIC_RECORD_SCHEMA_VERSION` 3→1, `TRACE_SCHEMA_VERSION` 4→3,
  `HUMANEVAL_METRICS_PROFILE_VERSION` "v1"→"0", scoring "v3"→"0".
- Contract conflicts: candidate-id (content-hash) dedup vs main's
  positional candidate identity; `parse_humaneval_dataset` has no
  `overrides=` kwarg; `extract_metrics` no longer performs
  procedure↔extraction cross-validation or per-trace producer admission.
- Gap in main surfaced by the audit: `load_humaneval_snapshot_rows` does
  no content-digest verification (branch verified snapshot SHA-256).

## #69 — Dynamic preprocessing analysis

- Executor: none. Only subprocess use is build-time npm/vite in
  `scripts/build_viewer_assets.py`.
- Dead dependencies: all corpus reads flow through #68 files. Own files
  import neither `dr_code.trace` nor `dr_code.metrics`; coupling to
  records is string-level SQL (`record_status = 'measured'`,
  `outcome = 'passed'`).
- Trace-shape breaks in `preprocessing_artifacts.py:614–650`: iterates
  `zip(output.candidates, output.lineage, strict=True)` and reads
  `origin.path` / `operation.kind` / `operation.details` — none exist on
  main (`candidates` is `tuple[CodeCandidate, ...]`; origins nested;
  `ExtractionOperation` has `operation_name` only). `candidate_id` is the
  primary key of `CANDIDATES_SCHEMA` and every downstream join; no
  main-side source exists.
- Silent-zero hazard: step-name and fact-key literals inside DuckDB
  `EXISTS` subqueries (`require_nonblank_text`, `'$.is_nonblank'`,
  `'$.candidate_count'`, `'$.survivor_candidate_count'`) match zero rows
  against a main-produced corpus; queries return empty results, not
  errors. `definition_version: str` admits "v1"-vs-"0" corpora without
  error.
- Survives nearly as-is: `viewer/{assets,database,app,cli}.py` (~2,700
  lines; imports nothing outside `dr_code.viewer`) plus the frontend. The
  #68 weld is 3 import statements (`viewer/domain.py:10–13`,
  `viewer/analytics.py:20–24`) plus one `validate_preprocessing_relations`
  call. `RunDescriptor` lives in `dr_code.corpus` and drags a 10-module
  closure.
- Re-derive: `run_descriptor.py`, `preprocessing_analysis.py`,
  `preprocessing_comparison.py` (~3,150 lines), `analytics.py` stage
  vocabulary (rebuild from `StepName`, pin literals with a golden test).

## #70 — Deterministic behavioral mutants

- Executor: oracle built on `dr_code.execution.subprocess` (never merged).
  Runs `sys.executable -I -c` on the host; no sandbox. Interface deltas vs
  main's seam: `input_text=`→`input_json=`; `SubprocessError` taxonomy vs
  `SandboxError`/`SandboxTimeoutError`/`SandboxOutputLimitError`;
  `SubprocessCompletedProcess` and `SandboxCompletedProcess` are
  structurally identical, nominally distinct — an import swap type-checks
  while changing the threat model.
- Provenance is executor-specific: `capture_production_runner` hashes host
  `sys.executable` and binds `["-I", "-c"]`; under the sandbox this
  attests to an interpreter that did not execute the program. A rename
  rebase produces datasets with false `runtime_identity`.
- Dead dependencies: `dr_code.eval.identity.identity_hash_for` (dataset,
  provenance), `dr_code.implementation_identity.package_source_digest`,
  `dr_code.corpus.{atomic_directory,stable_files}` (two symbols, ~170
  lines, severable), `HumanEvalSource` packaged-snapshot default (main's
  loader is explicit-path only).
- Survives verbatim: `operators.py` (332 lines, stdlib-only AST mutation),
  `outcomes.py` (46 lines), `seeded_site_order`, `evaluate_gates` /
  `distinct_input_indices`, the FD-duplication wire-protocol design
  (same idiom as main's `sandbox_runner_script.py`).
- Chain dependence: none on #69; #68 limited to the two filesystem
  primitives. Re-derivation estimate: ~600–800 source lines against 2.7k.
- Cache interaction: the determinism gate re-executes each mutant; routed
  through `run_requests` caching, the second run is a cache hit unless the
  gate bypasses the cache.
- Standalone decision surfaced: the branch relocates the 11 MB HumanEval+
  snapshot into the wheel (`src/dr_code/synthetic/humanevalplus_snapshot.json`,
  regenerated bytes) and makes packaged-offline the loader default.

## #71 — Protected task annotations

- Executor: none. All execution references in the diff are context lines.
- Provenance of the 35 files: 31 created by #69, 1 by the rejected eval
  kernel (`tests/eval/test_lifecycle_identity.py`), 2 pre-stack READMEs,
  4 new (`test_task_annotations.py`, `task-annotation.tsx`,
  `use-autosave-queue.ts`, `task-annotation.test.tsx`).
  `git apply --3way --check` onto main: 33/35 `does not exist in index`,
  2/35 conflict. #70 dependence: none.
- `task_identity` = `identity_hash_for` over
  `("task_id", "prompt", "canonical_solution", "entry_point", "test")`.
  Main's `HumanEvalTask` carries exactly those five fields; #90's
  notes-tuple and always-recompute changes do not perturb the payload.
  Main has no identity-hash primitive and no `dr_serialize` dependency.
- Survives as re-derivation source: annotation domain in `domain.py`
  (`TaskIdentity`, `TaskAnnotation`, `TaskAnnotationProvenance`, hardened
  JSON codecs; human ⇒ no provenance, machine ⇒ provenance required,
  enforced in `__post_init__`); `use-autosave-queue.ts` and
  `task-annotation.tsx` (self-contained); `test_task_annotations.py`
  (1,077 lines) as executable specification.
- DuckDB layer (`registered_tasks`, `task_annotations`,
  `task_annotation_tags`, four `archived_*` mirrors, fcntl owner lock)
  presupposes #69's `ViewerDatabase`.

## #72 — Failure classification

- Increment split: taxonomy/aggregation/prompt 539 lines; input extraction
  ~830; `lane.py` LLM-provider transport 1,679; persistence/publication
  2,688; CLI 283; tests ~5,600.
- Executor: classification reads persisted Parquet via `ViewerAnalytics`;
  no execution of candidate code. The single `dr_code.execution` import
  (`lane.py:25`) shells out to the LLM provider CLI — subject of the PR's
  own F72-1 deferral (provider settings and cwd state not bound into lane
  policy identity until after dr-exec).
- Taxonomy is self-contained: 2 families, 10 parse + 6 test labels,
  LLM-assigned; zero references to any preprocessing failure code.
  `failure_code` and `outcome` are passed through as opaque evidence,
  never matched.
- Dead dependencies: `dr_code.viewer` (#69), `dr_code.corpus.run_descriptor`
  (#68/#69), `identity_hash_for` (#66 kernel; thin wrapper over
  `dr_serialize`, re-pointable), #71's `TaskAnnotation` upsert machinery
  (`classify.py:1005`), the Parquet/DuckDB schema (absent from main
  entirely).
- Silent-wrong hazards: SQL literal `require_nonblank_text` (main:
  `reject_blank_input`) — rebases cleanly, matches zero rows, reports an
  empty parse-failure population; `'passed'`/`'measured'` are bare strings
  coinciding with `SubmissionOutcome.PASSED` / `RecordStatus.MEASURED`
  values, unpinned.
- Unassigned outcomes: `EXTRACTION_FAILED`, `EMPTY_SUBMISSION`,
  `EVALUATION_INCOMPLETE`, `NO_TOP_LEVEL_FUNCTIONS` (preempted at
  extraction on main) map to no classification family.
- Field-shape drift: evidence contract projects `best_function_name` and
  `coverage_complete` as stored columns; on main both are
  serialization-excluded computed fields (`EvaluationTaskSummary` is the
  boundary shape). `function_count` has no main equivalent.
- Lands now without chain dependencies: `taxonomy.py`, `aggregation.py`
  (reconcile against `dr_code.evaluation.aggregation`), `prompt.py`
  templates with render signatures re-derived against main's types
  (~550 source lines + 267 test lines).

## Cross-cutting decisions

1. Persistence layer: whether main adopts the #68 Parquet/DuckDB corpus
   substrate; gates 68→69→71→72 publication infrastructure.
2. Identity-hash primitive: absent from main; consumers in 68
   (coordinates), 70 (datasets), 71 (`task_identity`), 72 (taxonomy
   hashes). Options: vendor a canonical helper or pin `dr_serialize`.
3. Candidate identity in persisted artifacts: #68/#69 assume
   content-hash `candidate_id`; main's contract is exact source plus
   position in the materialized set.
4. Snapshot policy: packaged wheel resource with offline default (#70)
   vs explicit-path loader (main); byte-digest verification of the
   snapshot exists on the branches only.
5. Executor identity in persisted hashes (`RUNNER_IDENTITY`, mutant
   `runtime_identity`): convention set once, at dr-exec adoption.
