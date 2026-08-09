# Evaluation core foundation

Status: design brief

## Purpose

Define the context, ownership boundaries, required outcomes, and unresolved
questions for the next `dr-code` evaluation-core design effort. This document
is intentionally not an implementation plan. The implementing agent and
repository owner will refine the domain model, select the public contracts,
decide the delivery sequence, and produce the implementation plan together.

## Current context

`dr-code` main includes the async storage cutover from pull request 109 at
commit `9e23c1ea`. The repository now pins:

- `dr-exec==0.1.7`;
- `dr-store==0.2.0`; and
- `dr-serialize==0.1.2`.

The current repository already owns substantial evaluation-domain behavior:

- complete preprocessing traces and serialized trace evidence;
- typed metric definitions, questions, records, and aggregation outcomes;
- evaluation plans, task selections, and repeat plans;
- HumanEval parsing, execution semantics, and submission scoring;
- scoped preprocessing and execution caches; and
- closed attribution of candidate behavior versus evaluation-infrastructure
  failure.

The dependency cutover made cache and batch-scoring entry points asynchronous,
but it did not complete the broader evaluation core. In particular:

- execution cache misses still pass serially through a synchronous process
  call inside the async metric engine;
- `dr-code` still constructs raw Python driver jobs instead of using the
  released importable-JSON job capability;
- the current `SampleCoordinate` represents a planned task/repeat slot rather
  than an observed dataset sample with a stable sample identity;
- complete run observations are not represented by one typed persisted model;
- immutable evaluation-result bundles and their readers do not exist; and
- comparative replay and structural result comparison remain script-owned or
  absent.

Several open `dr-code` branches exercise adjacent behavior. Their assumptions
need to be considered during design rather than copied into the core:

- pull request 103 contains a HumanEval verification workflow with its own
  task-level concurrency and flattened result publication;
- pull request 108 contains generation-corpus models and a corpus-specific
  publication implementation, including stable `generation_id` values;
- pull requests 106 and 107 contain comparison and corpus-publication behavior
  that may consume the resulting evaluation contracts.

## Available foundations

### `dr-exec`

The released package owns bounded local process execution, asynchronous
resident-capacity scheduling, importable JSON jobs, trusted and untrusted
target declarations, finite job and protocol budgets, typed execution
outcomes, and backend-neutral run-record references.

An importable job is one process-isolation, cancellation, failure, and
recording unit. A finite application-owned batch may share that unit only when
its members intentionally share the same fate and evidence. Standalone fan-out
uses `ExecutionPool`; a durable workflow normally invokes one job for its own
recovery unit rather than nesting another scheduler.

### `dr-store`

The released package owns asynchronous content-addressed records and caches,
SQLite and PostgreSQL backends, and synchronous terminal artifact-bundle
publication and verified reading. Its generic bundle manifest owns artifact
names, byte lengths, hashes, and completeness while leaving the manifest
payload and artifact semantics to `dr-code`.

Artifact bundles are local directory publications intended at task, run, or
result grain. They are not per-event records, a packed execution-record store,
or a remote blob system. Async applications offload a complete bundle
operation rather than splitting its filesystem lifecycle across event-loop
tasks.

### Adjacent foundations

`dr-providers==0.3.0` owns provider-call requests, invocation evidence,
classification, retry policy, and complete logical-call results.
`dr-platform` is completing an async-stage and run-fan-in release that owns
durable linear scheduling, ordered run membership, membership digest binding,
and one run-completion barrier.

These packages inform the integration boundary, but their domain behavior does
not belong in `dr-code` core.

## Ownership boundary

`dr-code` owns:

- source preprocessing and candidate materialization;
- code-execution requests and interpretation of execution outcomes;
- metric extraction and evaluation-domain aggregation;
- evaluation identities and typed observations;
- immutable evaluation evidence and derived evaluation projections; and
- structural comparison of evaluation evidence.

`dr-code` should build on `dr-exec`, `dr-store`, and `dr-serialize` rather than
reimplement their process, scheduling, storage, hashing, or bundle primitives.

Whetstone and other experiment applications own:

- provider calls and model-generation lifecycle;
- prompt, reward, optimizer, and statistical-analysis policy;
- experiment membership and train/validation/test policy;
- `dr-platform` pipeline declarations and resource composition;
- application result bindings used by run fan-in; and
- experiment-level acceptance and baseline-versus-optimizer claims.

`dr-platform` owns durable scheduling and run completion without interpreting
evaluation inputs or outputs. `dr-providers` owns provider lifecycle without
knowing code-evaluation semantics. The design should preserve these boundaries
unless a concrete requirement proves that a dependency must move.

## Requested outcomes

The completed evaluation core should provide the following capabilities as one
coherent design.

### Canonical bounded execution

There is one `dr-code` execution path over the released `dr-exec` primitives.
It supports bounded concurrent standalone evaluation and a bounded single-unit
composition suitable for a durable application stage. Execution evidence
retains enough identity and record information to be audited independently of
derived metric values.

The boundary continues to distinguish candidate-attributable outcomes from
harness, runtime, and executor failures. It must not imply an operating-system
sandbox that the selected runtime does not provide.

### Correct evaluation identities

Planned evaluation slots and observed samples are distinct concepts. Persisted
candidate identity composes the actual sample, the resolved preprocessing
definition, and the post-materialization candidate position without treating a
content hash as semantic identity.

The design needs to work for both live generated samples and frozen historical
generation corpora. Existing stable generation identities, task identities,
task-selection order, repeat structure, and provenance should compose without
heuristic reconstruction.

### Typed complete observations

Authoritative evidence preserves the existing typed domain contracts rather
than reducing them immediately to analysis rows. The evidence model covers the
selected raw input, preprocessing trace, materialized candidates, metric
records, and execution evidence, including zero-candidate, inapplicable,
candidate-failure, and infrastructure-failure cases.

Every selected input remains attributable through the pipeline even when it
does not produce a measured candidate result. Batch behavior makes partial
completion and invalid-run conditions explicit rather than silently dropping
failed members.

### Immutable evaluation bundles

`dr-code` defines a versioned evaluation-specific manifest payload and artifact
schemas over the released `dr-store` artifact-bundle envelope. The typed source
evidence remains authoritative; candidate-, generation-, task-, and other
analysis tables are reproducible projections.

Bundles contain portable references and relative artifact names rather than
machine-specific paths. Readers validate the generic integrity envelope and
the evaluation-specific payload before interpreting artifacts.

### Reusable batch evaluation

One reusable batch boundary composes preprocessing, candidate materialization,
cache prefetch, bounded execution, metric production, evidence publication,
and derived summaries. Callers do not need private thread pools, duplicate
cache layers, or script-specific execution loops to obtain useful throughput.

The same evaluation semantics apply whether the work is run locally as a
bounded batch or divided into durable application work items. Concurrency and
recovery granularity must not alter score meaning, ordering identities,
denominators, or failure attribution.

### Structural replay and comparison

Persisted evidence supports replay from frozen raw generations and, where the
available evidence permits, from frozen materialized candidates. Comparison
can identify changes in selected samples, preprocessing results, ordered
candidates, metric records, and candidate-, generation-, and task-level
aggregates while reporting the relevant denominators.

Benchmark-specific acceptance and statistical claims remain outside this
structural comparison layer.

## Existing guarantees to preserve

The design must remain consistent with the repository's authoritative terms
and contracts, especially:

- traces are complete defensive snapshots;
- serialized traces and metric records are self-describing;
- preprocessing distinguishes declared absences from unexpected exceptions;
- metric outcomes distinguish measured, not-applicable, and operator-failure
  records;
- evaluation plans are internally complete;
- aggregation is deterministic over explicit slots;
- HumanEval candidate behavior is not relabeled as harness failure, and
  harness failure is not relabeled as candidate behavior;
- every execution job has finite enforced budgets and a fixed environment;
- persistent caches remain scoped, explicit, and best effort; and
- development changes use a hard cutover rather than maintaining parallel old
  and new execution or persistence paths.

Changes to repository vocabulary, public models, persisted schemas, or these
standing guarantees require corresponding updates to `.defs/terms.toml`,
`.defs/contracts.toml`, public exports, documentation, and contract evidence.

## Design questions for owner collaboration

The design phase should resolve these questions with the repository owner
before freezing public or persisted contracts:

- What is the intended process-job and failure-fate grain for HumanEval and
  other future evaluators?
- Which execution configuration belongs to the reusable evaluation boundary,
  and which belongs to standalone or durable application composition?
- How should a planned slot, an observed live sample, and a historical
  `generation_id` relate without introducing optional or ambiguous identity?
- What are the authoritative observation envelopes, and where should
  references be used instead of embedding complete evidence?
- Which artifacts belong in an evaluation bundle, which metadata belongs in
  its manifest payload, and which tables are derived projections only?
- What batch failure policy distinguishes invalid evaluation runs from valid
  runs containing candidate failures or explicit missing observations?
- How are runtime identity, cache scope, execution records, and replay mode
  represented together?
- What structural matching rules make comparisons truthful when candidates
  are added, removed, reordered, or changed?
- Which parts of the open verification, corpus, and comparison branches should
  consume the new core, and what landing order minimizes conflicting contract
  definitions?
- Which local filesystem evidence belongs in terminal bundles and which
  high-volume or cross-worker records belong in asynchronous record storage?

These are design inputs, not implied selections. The implementation plan should
follow only after the selected answers form one internally consistent model.

## Out of scope

This effort does not add:

- provider-specific clients, retry rules, or provider-call orchestration;
- prompt rendering, model generation, rewards, optimization, bootstrap power
  estimation, or experimental split policy;
- a `dr-platform` scheduler, pipeline, or fan-in implementation inside
  `dr-code`;
- a generic experiment framework or arbitrary workflow graph;
- another content-addressing, canonical-JSON, hashing, process, scheduler,
  cache, or generic artifact-bundle implementation;
- an operating-system sandbox claim; or
- compatibility shims preserving superseded development APIs or persisted
  schemas.

## Completion criteria

The design and later implementation are complete when:

- one documented evaluation model covers live and historical inputs without
  conflating planned and observed identity;
- all execution uses the selected canonical `dr-exec` path with explicit
  bounds and preserved attribution;
- standalone callers can evaluate a bounded batch without private concurrency
  machinery;
- durable applications can invoke the same evaluation semantics without
  importing application policy into `dr-code`;
- authoritative typed evidence can be published, verified, restored, replayed,
  and structurally compared;
- derived tables and summaries can be reproduced from that evidence;
- adjacent open branches have an explicit alignment and migration disposition;
  and
- the repository's public API, terms, contracts, tests, documentation, and
  installed-wheel behavior agree with the selected design.
