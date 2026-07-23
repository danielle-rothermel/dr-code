# Rebase guide: PR 57 onto `impl/01-eval-kernel`

## Purpose

PR 57 and its lower stack build the preprocessing, corpus, candidate-evaluation,
analysis, and viewer workflow that is currently being used to inspect parser
behavior. `impl/01-eval-kernel` is the main working copy used to test and debug
Whetstone AI evaluation changes. The intended integration is not a winner-takes-
all merge: it must retain the evaluation kernel's lifecycle and identity model
while preserving the preprocessing stack's exhaustive extraction, diagnostics,
artifact production, HumanEval execution behavior, analysis, and viewer.

This document records the exploratory comparison and gives an ordered integration
strategy. It does not perform or authorize the rebase itself.

## Current divergence

The references at the time of this analysis are:

| Reference | Commit |
| --- | --- |
| Merge base | `666bcde89ba6061121d12665589f45c35ac34fc0` |
| `impl/01-eval-kernel` | `9a597c672eccec3b311ec96c17323bee4fe42a9a` |
| PR 57 branch (`07-19-preprocessing_viewer`) | `3798dc03a5497afbfb327aa864a4e80cef8a8df5` |

The branches contain three eval-kernel-only commits and seven PR-stack-only
commits after the merge base.

Eval-kernel-only commits:

1. `0deb87d` adds a temporary editable `dr-serialize` dependency.
2. `a10d65c` implements the evaluation kernel.
3. `9a597c6` excludes RNG seeds from `RepeatPlan` identity.

PR-stack-only commits:

1. `47984d7` adds analysis-grade preprocessing trace contracts.
2. `73fd0b7` builds exhaustive function-candidate preprocessing.
3. `6c37342` adds resumable corpus artifacts.
4. `d5008e6` adds candidate evaluation and preprocessing analysis.
5. `302509a` adds the all-failures preprocessing explorer.
6. `2a70190` plans the dynamic preprocessing viewer.
7. `3798dc0` builds the dynamic preprocessing viewer.

The main conclusion is that the branches are not far apart mechanically, but
they are meaningfully different at their domain boundaries. A trial merge
predicts direct textual conflicts only in `pyproject.toml` and `uv.lock`.
That understates the work: independently added types and identity rules can
merge cleanly while still representing the same concept differently.

## Mechanical conflicts

### Dependencies

The resolved dependency set must retain all of the following:

- eval kernel: `dr-serialize` and its source declaration;
- preprocessing/corpus: `pyarrow`;
- dynamic viewer: `duckdb`, `fastapi`, `uvicorn[standard]`, development
  `httpx`, and the `dr-code` command entry point.

Resolve `pyproject.toml` intentionally, then regenerate `uv.lock` with `uv`.
Do not combine lockfile conflict hunks by hand.

The current `dr-serialize` declaration is explicitly marked as a temporary
editable path dependency. A published or otherwise stable source is required
before the integrated branch can be treated as merge-ready.

### Generated artifacts

PR 57 contains checked analysis outputs produced under the preprocessing
stack's current identities and schemas. Those outputs must not be silently
presented as current after identity or record-shape changes. Either regenerate
them after integration or retain them behind an explicit legacy schema reader.
The dynamic viewer has already removed its copied static data and queries the
authoritative Parquet artifacts directly.

### Host subprocess execution

The host subprocess backend is an intentional PR 57 contract, not a conflict
resolution detail. A rebase onto `impl/01-eval-kernel` must not restore the
older Docker/OCI runner, its image configuration, or per-evaluation container
startup. Preserve all of these coordinates together:

- `dr_code.humaneval.subprocess_runner` as the production execution boundary;
- `run_python_subprocess` as the default runner and `run_in_subprocess` as the
  injection keyword at scoring and evaluation boundaries;
- a fresh `[sys.executable, "-I", "-c", source]` child for each request;
- bounded JSON input and output, finite positive deadlines, and process-group
  termination on timeout or output overflow;
- the minimal child environment, without inherited credentials;
- the exact `subprocess:python-isolated@v1` runner identity; and
- `sandbox_image: null` only as a legacy manifest-schema field.

There must be no image pull, image preflight, container-runtime environment
variable, or Docker/Podman command in production evaluation or CI. Resolve any
rebase conflict in favor of the subprocess backend, then run the focused
contract tests before continuing the semantic integration.

This backend is not an operating-system sandbox. The integrated workflow must
continue to document that model-generated code can use the host permissions of
the evaluation worker and therefore belongs only on a disposable, constrained
host.

## Semantic conflict map

### Preprocessing definitions and configuration

The stack has operational preprocessing definitions in
`dr_code.preprocessing`, while the eval kernel introduces lifecycle-facing
`PreprocessingDefinition` and materialized configuration concepts in
`dr_code.eval`. These should not remain two permanent authoritative public
models.

Recommended direction: make the eval lifecycle/configuration model canonical,
or introduce a temporary, lossless, tested adapter from the operational model.
The operational runner must retain ordered executable steps, unique instance
names, settings validation, resolved implementation versions, and reserved-name
validation.

### Identity and hashing

The preprocessing stack currently uses BLAKE2-based `stable_hash`,
`preprocessing_definition_hash`, `metrics_definition_hash`, a custom task
fingerprint, and a custom evaluation-key payload. The eval kernel uses canonical
SHA-256 identities from `dr-serialize`, plus explicit task, configuration,
procedure, and repeat identities.

New persisted artifacts should use the eval kernel's canonical identities. The
old hashes may be supported only as versioned legacy coordinates; they should
not remain a second source of truth. Candidate execution identity should be
built from:

- HumanEval task identity;
- candidate content identity;
- metric and procedure configuration identities;
- the explicit execution/runtime fingerprint.

RNG seed exclusion from `RepeatPlan` identity must remain intact.

### Candidate and trace models

Both branches define code artifacts and candidate-set concepts. PR 57 adds
stable candidate ordering, content-derived IDs, multiple extraction origins,
compile diagnostics, rejections, and top-level-function information. The eval
kernel adds typed, compile-validating code and candidate artifacts.

The integrated candidate model must preserve all of PR 57's diagnostic and
lineage data. In particular, a single `origin: str` is not sufficient to retain
multi-origin extraction lineage.

There is also a material semantic difference around zero candidates:

- PR 57 currently represents some all-filtered cases as `Absent` and records a
  terminal failure cause.
- The eval kernel distinguishes a valid empty candidate set from a causal
  preprocessing failure.

Recommended direction: preserve an explicit empty candidate set when processing
successfully yields zero candidates, while retaining terminal reason facts and
rejections for analysis. Reserve `Absent` for actual causal failures. Tests must
distinguish missing input, missing trace/key, preprocessing failure, and a valid
empty result.

### Metrics, facts, and HumanEval evaluation

The branches contain overlapping metric-definition and `MetricRecord` families.
PR 57's HumanEval path directly produces candidate outcomes and a binary score;
the eval kernel models neutral `MetricFact`s, operator/procedure lineage,
`MetricRecord`s, `Score`s, and explicit aggregation policy.

Recommended direction:

- represent `CodeTestResult` fields as eval-kernel facts with units and operator
  lineage;
- derive eval-kernel records and scores from those facts;
- retain PR 57's outcome view as a higher-level policy/export view;
- use explicit aggregation configuration where descriptive facts become rates
  or scores, including missing-value and zero-denominator behavior.

PR 57's execution cache, evaluation deduplication, infrastructure-failure
separation, subprocess startup and process-group cleanup, and bounded-I/O
behavior are additive and should be retained.

### Task and repeat provenance

Use `humaneval_task_identity` and a materialized `TaskSet` instead of the current
full-model task fingerprint. The source corpus currently guarantees only
`sample_id` and `task_id`; faithfully constructing the eval kernel's
`RepeatPlan` may require adding repeat index or seed provenance to the generation
corpus contract.

### Persisted corpus and evaluation schemas

Corpus and evaluation manifests currently embed the preprocessing stack's old
hashes and record shapes. Changing canonical identity, empty-set semantics, or
metric records requires a new artifact schema version. Do not write the changed
shape while continuing to label it schema version 1.

New producer metadata should include the eval definition/configuration identity,
resolved versions, procedure lineage, and runtime coordinate while preserving
the stack's relational integrity checks, atomic writes, resumability, and trace
facts.

### Analysis and dynamic viewer

The viewer should remain a thin consumer of versioned artifact contracts. Its
annotation identity—corpus SHA-256, sample ID, and exact decoder-output
SHA-256—should be preserved.

Before/after preprocessing comparison must allow different preprocessing
configuration hashes when both runs use the same corpus and expose a compatible,
validated stage contract. Evaluation-stage comparison additionally requires
compatible task, metric, procedure, and execution coordinates. Viewer failure
queries will need to understand the integrated empty-candidate representation
without losing PR 57's terminal failure grouping.

## Intention-preservation map

| PR 57 intention | Integration landing point |
| --- | --- |
| Ordered executable preprocessing with validated settings | Eval lifecycle definition/config, consumed directly or through a lossless adapter by the operational runner |
| Step or operator behavior changes identity | Resolved versions and canonical `dr-serialize` configuration hashes |
| Exhaustive, stable candidate extraction | Eval candidate set, preserving position, IDs, and all extraction origins |
| Detailed terminal failures and rejections | Versioned trace facts and rejection relations alongside valid empty sets |
| Full serializable traces | Existing trace schema and migration behavior, extended with eval config/procedure lineage |
| Atomic, resumable Parquet corpus processing | Existing corpus runner with versioned eval identities in its manifest |
| Deduplicated candidate execution | Eval task/config/procedure identities plus candidate content and runtime fingerprint |
| Pinned HumanEval snapshot | Eval `TaskSet` and `humaneval_task_identity` |
| Test diagnostics and candidate outcomes | Eval facts/records/scores, with PR 57 outcomes retained as a policy view |
| Analysis tables and rates | Direct descriptive analysis plus explicit eval aggregation where reduction policy matters |
| Dynamic viewer and annotations | Versioned artifact adapter with the existing annotation key |
| Before/after comparison | Equal corpus plus compatible stage contracts, not equal preprocessing hashes |
| Host subprocess execution | Preserve the subprocess API and exact `subprocess:python-isolated@v1` runtime coordinate; never restore Docker/OCI during conflict resolution |

## Superseded and additive work

Likely superseded as authoritative contracts:

- the preprocessing and metrics hash helpers for newly written artifacts;
- the custom task fingerprint and evaluation-key payload;
- one of each duplicate preprocessing-definition, metric-definition,
  metric-record, and candidate model families;
- direct scoring records where eval facts, records, scores, and lineage become
  canonical;
- checked static viewer JSON and synchronization machinery, which PR 57 has
  already removed.

Additive work to preserve:

- exhaustive extraction variants and strategies;
- candidate deduplication, stable ordering and IDs, multi-origin lineage,
  rejections, compile warnings, and top-level-function filtering;
- bound preprocessing runner reuse;
- trace serialization, finite JSON fact validation, and legacy trace migration;
- atomic/resumable corpus artifacts and relational integrity validation;
- leased candidate evaluation, execution deduplication, and cache behavior;
- bounded host-subprocess execution and separation of planned requests from
  execution;
- analysis reports and tables after their identities/schemas are updated;
- the DuckDB/FastAPI/React viewer and durable annotation workflow;
- all eval-kernel lifecycle, identity, task/repeat, fact, aggregation, and
  compression-reference behavior.

## Recommended integration sequence

1. **Freeze a verified PR 57 head.**
   Use `3798dc0` or a later explicitly verified documentation-only head as the
   source of the rebase. Keep the dynamic viewer plan and this guide with it.

2. **Replay the seven PR-stack commits onto `impl/01-eval-kernel`.**
   Resolve only actual textual conflicts at this stage. Retain all dependencies
   in `pyproject.toml` and regenerate `uv.lock`.

   Gate: dependency resolution succeeds under `uv run --frozen`, and the eval
   kernel's existing tests still pass.

3. **Establish one identity and model crosswalk.**
   Decide the canonical definition, configuration, metric, record, code, and
   candidate types before modifying persisted artifact writers. Prefer the eval
   kernel models, extended where needed to carry PR 57 information, over two
   permanent parallel models.

   Gate: eval golden identity tests remain stable except for deliberately
   reviewed extensions; operational-to-eval conversion has golden tests.

4. **Reconcile preprocessing and candidate semantics.**
   Preserve extraction behavior, ordering, IDs, origins, diagnostics, and
   rejections. Adopt the eval kernel's distinction between empty results and
   causal absence without losing terminal analysis facts.

   Gate: all preprocessing and trace tests pass, including explicit missing,
   failed, empty, and successful candidate-set cases.

5. **Version corpus artifacts and manifests.**
   Add canonical eval identities and procedure coordinates, update result
   derivation, and retain deterministic, atomic, resumable output.

   Gate: interruption/resume, incompatibility detection, relational integrity,
   deterministic export, legacy migration, and a small golden corpus pass.

6. **Move HumanEval candidate evaluation onto the kernel.**
   Use task-set and metric/procedure identities; emit eval facts, records, and
   scores. Preserve execution leasing/cache behavior, infrastructure failures,
   subprocess behavior, and candidate membership.

   Gate: code-test parity, execution deduplication, task/evaluation identities,
   failure classification, subprocess cleanup, and bounded-I/O tests pass. The
   production and operational contract contains no Docker/Podman invocation or
   runtime/image requirement.

7. **Integrate aggregation and analysis.**
   Keep neutral descriptive facts and make reduction policy explicit for rates
   and scores. Update analysis readers for the versioned schemas.

   Gate: analysis totals reconcile to source relations, including missing and
   zero-denominator cases.

8. **Adapt the dynamic viewer last.**
   Update run registration, stage compatibility, evaluation coordinates, and
   empty-candidate failure queries after the underlying artifact contract is
   stable.

   Gate: backend tests, frontend typecheck/tests/build, exact aggregate-to-list
   drilldowns, deterministic annotation export, and real-artifact reconciliation
   pass.

9. **Regenerate or archive old outputs.**
   Do not mix legacy and integrated identities under the same schema label.

10. **Run the complete matrix.**
    Run `uv run --frozen pytest`, Ruff, Ty, all frontend workspace tests,
    frontend typechecking/build, focused subprocess tests, and at least one
    interrupted/resumed corpus and candidate-evaluation run.

## Decisions required before the semantic integration

1. What published or stable source replaces the temporary editable
   `dr-serialize` dependency?
2. Which eval-kernel model is canonical for preprocessing definitions,
   candidates, metric definitions, and records, and which PR 57 fields must be
   added to make that model lossless?
3. Will old corpus/evaluation artifacts be regenerated, or will a versioned
   legacy reader be maintained?
4. What repeat index or seed provenance must be added to the generation corpus
   to materialize `RepeatPlan` faithfully?
5. Do any external consumers require compatibility for the parser/scoring APIs
   PR 57 intentionally replaced? Compatibility aliases should not be added
   without an identified consumer.

These decisions should be made before changing artifact writers. Making them
afterward would risk a second identity migration and another regeneration of the
large corpus outputs.
