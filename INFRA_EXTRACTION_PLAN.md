# Executor-independent infrastructure extraction

Four stacked PRs off `main` landing the executor-independent infrastructure
from the rebuild stack. This doc rides in PR 1 and is removed in PR 4.

Endgame: PRs 66/67 are closed and re-cut minimal (metrics engine execution,
`code_test` operator, HumanEval scoring/task) once dr-exec is ready; PR 64 is
superseded by dr-exec adoption.

## PR 1 — Trace v3

- `Trace` construction snapshots `values` and deep-copies `step_facts`;
  `JsonArtifact.payload` is copied at validation, so the snapshot claim
  covers the containers, the step facts, and artifact payloads.
- Step facts widen from `Mapping[str, str]` to validated finite JSON:
  string keys, finite floats, no cycles, no non-JSON values.
- `Absent` gains `failure_code: str` — a plain string at the trace layer;
  failure vocabularies are owned by producers. `cause` and
  `propagated_through` remain.
- `CodeCandidateSetArtifact.candidates: tuple[str, ...]` is replaced by
  nested records: `ExtractionOperation`, `CandidateOrigin`, `CodeCandidate`
  (source + one or more origins).
- New inspected-candidate artifact: `CandidateInspection`,
  `InspectedCodeCandidate` (candidate + inspection),
  `InspectedCodeCandidateSetArtifact`. Inspection fields are structural only
  (parses, compiles, error text, top-level function names) — no policy
  verdicts.
- Candidate identity within a trace is exact-source equality plus position;
  no hashes or source-derived semantic IDs.
- Trace schema version bumps 2 → 3. No v2 archive model.
- Not in the trace layer: `SampleCoordinate` / `CandidateCoordinate`,
  HumanEval acceptance policy, executor outcomes / return codes /
  subprocess facts, a generic immutable-JSON framework.

## PR 2 — Preprocessing hard cut

- Runner API: `bind_preprocessing(definition)` returns a
  `BoundPreprocessingRunner`; `BoundPreprocessingRunner.run(input)` performs
  the existing fold; `run_preprocessing(...)` is the one-shot wrapper.
  Explicit external binding path for unregistered definitions is preserved.
- Facade curated to: definition models, resolver, bound runner, binding
  functions, one-shot runner. `REGISTRY`, `Step`, `CandidateMapStep`,
  `BoundStep`, strategy registries, and individual step mechanics become
  private.
- The first-success strategy ladder and `select_first` definitions are
  replaced by one exhaustive function-candidate definition:
  1. Normalize text.
  2. Reject blank input explicitly.
  3. Extract candidates additively from all supported representations:
     raw response; fenced and unfenced segments; whole-response JSON
     strings; top-level JSON `code`; field markers; escaped-Python
     recovery.
  4. Apply candidate-local cleaning — including import repair, inference,
     and deduplication — while extending lineage.
  5. Add last-return salvage as an additional candidate (no destructive
     truncation of the original).
  6. Remove blank candidates.
  7. Deduplicate exact sources, merging their origins.
  8. Parse and compile each distinct source once to produce its stored
     inspection. Every source-mutating step precedes inspection, so an
     inspection always describes the exact source it accompanies.
  9. Apply plain-literal, code-representation, compilability, and
     top-level-function filters. Compilability and top-level-function
     checks read the stored inspection; the plain-literal and
     code-representation module-shape classifications derive from a
     memoized parse — `CandidateInspection` carries no filter-specific
     fields.
  10. Materialize the complete ordered candidate set as the final output.
- Removed: `AlternativesStep`, the public strategy registry, the
  best-effort and field-marker registered definitions, destructive
  `DropAfterLastReturn`, repeated-parse filtering and import-inference
  steps, `SelectFirst`, the redundant terminal `ReturnAll`.
- Failures use a closed `PreprocessingFailureCode` `StrEnum` (owned by
  preprocessing, stringified into `Absent.failure_code`) and may attach
  structured JSON evidence; the runner converts them into `Absent`
  preserving code, cause, facts, and propagation path.
- No hidden decoder normalization; lone surrogates remain a documented
  rejected input.
- `humaneval/code_extraction.py` and `code_parsing.py` cut over to the new
  pipeline in this PR; the legacy `CleaningTrace` / `ExtractionTraceNode`
  path is deleted; the first-success parity test
  (`tests/preprocessing/test_parity.py`) is retired.
- `candidate_ordinal` indexes the final materialized inspected-candidate
  set (post-dedup, post-filter); this definition is documented here and in
  the evaluation package (PR 4).

## PR 3 — MetricRecord evolution

- `MetricRecord` gains an explicit initial schema version.
- The anonymous `values` dict is replaced by ordered `MetricFact` models:
  fact name, strict finite scalar value, explicit unit. Every operator
  declares units.
- The three outcomes (measured, not applicable, operator failure) become a
  closed discriminated union instead of one model with nullable field
  groups.
- Not-applicable records nest the complete `Absent` object.
- Operator-failure records nest a small structured failure value.
- Records continue nesting the complete producer and metrics-definition
  coordinates; identity uses a question coordinate or nested declared
  question instead of independently flattened metric / target key /
  settings.
- Records validate without the live registry: the `resolve_settings_model`
  registry lookup at deserialization time is removed so archived records
  stay loadable after registry changes.
- No `Applicability` / `AbsenceMode` enums; no archived record model.

## PR 4 — Evaluation package

- New package `dr_code.evaluation`:
  - `DatasetCoordinate`.
  - `TaskSet` + `TaskSetCoordinate`: dataset coordinate, ordered source
    population, ordered selected task identities.
  - `RepeatPlan` + `RepeatPlanCoordinate`: uniform, contiguous, task-major
    repeat slots; optional seeds.
  - `SampleCoordinate`: task-set and repeat-plan coordinates + task
    identity + repeat index.
  - `CandidateCoordinate`: sample coordinate + preprocessing-definition
    coordinate + candidate ordinal (defined in PR 2).
  - `EvaluationProcedure`: resolved preprocessing and metrics definitions.
  - `AggregationPolicy`.
  - `EvaluationPlan`: task set + repeat plan + procedure + aggregation
    policy.
  - `Score`: name, strict scalar, unit, evaluation coordinate, source fact
    coordinates — derived value, not a verdict in metric facts.
  - Pure aggregation inputs/results and `aggregate(...)` taking a typed
    `AggregationPolicy`, distinguishing missing, not-applicable,
    zero-denominator, and non-finite results.
- No `SamplingPlan`.
- Excluded: Definition/Config pairing framework, variable templating or
  normalized-JSON assignment, semantic hashes, registry resolution inside
  generic evaluation values, executor requests or outcomes,
  HumanEval-specific task conversion or scoring policy.
- This PR removes `INFRA_EXTRACTION_PLAN.md`.
