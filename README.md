# dr-code

[![CI](https://github.com/danielle-rothermel/dr-code/actions/workflows/ci.yml/badge.svg)](https://github.com/danielle-rothermel/dr-code/actions/workflows/ci.yml)

[Terms and contracts](https://danielle-rothermel.github.io/dr-code/) ·
[terms source](https://github.com/danielle-rothermel/dr-code/blob/main/.defs/terms.toml) ·
[contracts source](https://github.com/danielle-rothermel/dr-code/blob/main/.defs/contracts.toml)

**Personally owned dependencies:**
[dr-exec](https://github.com/danielle-rothermel/dr-exec),
[dr-serialize](https://github.com/danielle-rothermel/dr-serialize), and
[dr-store](https://github.com/danielle-rothermel/dr-store).

**dr-code prepares, evaluates, analyzes, and visualizes Python code produced by
language models.**
The repository contains a Python library and a separately packaged React
viewer, organized into these functional areas:

- **[Candidate preparation](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/preprocessing)**
  turns raw model responses into inspected Python candidates through declared,
  ordered preprocessing operations.
- **[Caching](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/caching)**
  memoizes preprocessing traces and checkpoints reusable execution outcomes
  through caller-supplied record stores.
- **[Trace capture](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/trace)**
  preserves intermediate artifacts, structured facts, failure reasons, and
  semantic provenance so results remain explainable and serializable.
- **[Measurement](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/metrics)
  and [evaluation](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/evaluation)**
  extracts typed measurements from traces, declares evaluation plans, and
  reduces complete measurement slots into typed aggregation outcomes.
- **[HumanEval+ evaluation](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/humaneval)**
  loads and samples benchmark tasks, extracts candidate solutions, runs them
  through a [dr-exec](https://github.com/danielle-rothermel/dr-exec)
  executor, and reports structured outcomes.
- **[Generation corpus extraction](docs/generation_corpus.md)**
  converts archived model activity into validated, content-addressed Parquet
  tables while preserving raw evidence, task material, prompts, requests, and
  configuration provenance at their natural grains.
- **[Synthetic dataset generation](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/synthetic)**
  applies deterministic corruption recipes to known solutions for
  preprocessing and robustness experiments.
- **[Code visualization](https://github.com/danielle-rothermel/dr-code/tree/main/viewer/packages/viewer)**
  provides reusable React components for highlighted code, diffs, and status
  presentation, plus a private gallery for visual development.
- **Infra**
  - **[Core models](https://github.com/danielle-rothermel/dr-code/blob/main/src/dr_code/core/models.py)**
    provide frozen boundary models shared by the functional packages.
  - **[Source](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/core/source)**
    provides shared Python source inspection and transformation.
  - **[Execution](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/core/execution)**
    provides the shared dr-exec execution boundary.

## Functional areas

The sketches below show the current shape of the primary contracts. They are
abridged deliberately: `...` omits validators, defaults, derived fields, and
implementation details that belong in the linked package.

### [Candidate preparation](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/preprocessing)

Preprocessing is an ordered, versioned declaration of named steps. Binding
validates that declaration once; the resulting runner can then turn typed
input artifacts into complete traces.

```python
class StepSpec(FrozenModel):
    instance_name: str
    step: StepName
    settings: StepSettings = ...


class PreprocessingDefinition(FrozenModel):
    definition_id: str
    version: str
    steps: tuple[StepSpec, ...]
```

```python
@dataclass(frozen=True, slots=True)
class BoundPreprocessingRunner:
    definition: PreprocessingDefinition
    producer: TraceProducer
    ...

    def run(self, input_value: Artifact) -> Trace: ...


def bind_preprocessing(
    definition: PreprocessingDefinition,
) -> BoundPreprocessingRunner: ...
```

### [Batch preprocessing](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/caching)

Parallel batch preprocessing runs on dr-exec's worker pool. `preprocess_batch`
deduplicates the requested texts and runs each one as one trusted importable
JSON job across long-lived worker processes, which import the preprocessing
entry point once per worker.

```python
from dr_code.caching import preprocess_batch

traces_by_text = await preprocess_batch(
    texts,
    definition=definition,
    worker_count=16,
)
```

Pass `on_trace` to consume each result as it completes instead of retaining the
whole batch; the returned mapping is then empty.

```python
await preprocess_batch(
    texts,
    definition=definition,
    worker_count=16,
    on_trace=lambda text, trace: consume(text, trace),
)
```

A caller that wants candidate sources rather than whole traces uses
`candidate_sources_batch`, which runs an entry point returning the sources
alone. Result size is the term that decides what the pool costs: the caller
decodes and validates every byte a worker returns, single-threaded, so a whole
serialized trace costs about a hundred times the payload of the sources it
carries and no number of workers recovers that difference.

```python
from dr_code.caching import candidate_sources_batch

sources_by_text = await candidate_sources_batch(
    texts,
    definition=definition,
    worker_count=16,
)
```

### Execution primitives

Each kind of work in this repository runs on the dr-exec execution mode that
matches its trust boundary and its cost per item. See dr-exec's
[parallelism guide](https://github.com/danielle-rothermel/dr-exec#choosing-an-execution-mode-a-parallelism-guide)
for the full decision table.

| Work | Primitive | Why |
|---|---|---|
| Candidate and test execution | Spawned subprocess jobs (`ProcessExecutor`) | The source is model-produced and untrusted, so each candidate needs a real process boundary, an enforced budget, and a durable record of its own. |
| Preprocessing | dr-exec worker pool (`WorkerPoolImportableJsonExecutor`, via `preprocess_batch`) | Trusted first-party code, CPU-bound, milliseconds per item: long-lived workers pay the import once each and use real cores, where spawn-per-job would spend all of them on `import`. |

### [Windowed execution caching](docs/windowed_execution_cache.md)

`WindowedExecutionCache` bulk-prefetches planned execution observations into a
bounded resident window. It retains at most one bounded persistence batch in
flight and one bounded pending batch; further outcomes remain memory-only.
Normal close attempts one final checkpoint. Persistent read, validation, and
write failures are logged and degrade to misses or dropped retry state rather
than failing evaluation.

A candidate-owned execution observation becomes eligible for persistence once
the owning sample evaluation record has a portable evidence reference; harness
and infrastructure outcomes are never persisted. Batch assembly publishes that
reference with the observation, then releases the resident cache key; reuse
therefore never fabricates a source record or keeps attempt-wide pending cache
state. An evaluation batch request declaring itself `fresh` skips lookup for
its generations and re-executes every candidate, without replacing entries
already bound under the same keys.

Persistent keys combine the opaque execution-request key with the full digest
of a mandatory, caller-owned runtime identity. The identity must cover the
runtime, harness, dependency environment, and any other ambient behavior not
already represented by the request. The executor object itself is not part of
the persisted key.

```python
from dr_code.caching import WindowedExecutionCache
from dr_code.evaluation import EvalRuntimeIdentity
from dr_serialize import build_identity_document

runtime = EvalRuntimeIdentity(
    document=build_identity_document(
        schema="example/python-runtime",
        schema_version=1,
        payload={"python": "3.13.2", "environment": "experiment-image@sha256:..."},
    )
)

async with WindowedExecutionCache(
    batch_record_store,
    runtime=runtime,
    max_resident_entries=1_000,
    max_pending_checkpoint_entries=100,
) as cache:
    await cache.prefetch(planned_request_keys)
    ...
    cache.discard(assembled_request_key)
```

The injected store must provide async `get_many(keys, *, schema=...)`,
returning each distinct key as a verified hit or explicit `None` miss, and
atomic async `put_many(entries)`, where every entry carries its schema and
record. Point-only record stores are not adapted because per-entry persistence
would violate the bulk-I/O contract. Callers use persistent reuse only for
workloads whose outcomes they treat as stable within the runtime scope and
coordinate one active writer for that scope. Concurrent writers are unsupported
because this cache does not reconcile a different first-writer winner after a
checkpoint.

### [Trace capture](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/trace)

A trace is a stable snapshot of typed artifacts or explicit absences, together
with structured facts and the coordinate of the producer that made it. Public
reads are defensive projections, and persisted traces remain loadable without
consulting the current component registries.

```python
class CodeArtifact(FrozenModel):
    kind: Literal[ArtifactKind.CODE] = ArtifactKind.CODE
    source: str


TraceValue = Artifact | Absent


class Trace:
    def __init__(
        self,
        values: Mapping[str, TraceValue],
        producer: TraceProducer,
        step_facts: Mapping[str, Mapping[str, JsonFactValue]] = ...,
    ) -> None: ...

    @property
    def values(self) -> Mapping[str, TraceValue]: ...

    @property
    def step_facts(self) -> Mapping[str, Mapping[str, JsonFactValue]]: ...

    def value(self, key: str) -> TraceValue: ...
```

```python
class SerializedTrace(FrozenModel):
    schema_version: Literal[3]
    producer: TraceProducer
    values: dict[str, TraceValue]
    step_facts: dict[str, dict[str, JsonFactValue]]


def serialize_trace(trace: Trace) -> SerializedTrace: ...
def deserialize_trace(serialized: SerializedTrace) -> Trace: ...
```

### Measurement and evaluation

[`dr_code.metrics`](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/metrics)
asks versioned questions of trace values and returns one typed record per
question.
[`dr_code.evaluation`](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/evaluation)
composes preprocessing and metrics into a complete plan, then reduces explicit
measurement slots under a declared policy.

```python
class MetricQuestion(FrozenModel):
    metric: MetricName
    on: str
    settings: OperatorSettings = ...


class MetricsDefinition(FrozenModel):
    definition_id: str
    version: str
    questions: tuple[MetricQuestion, ...]


async def extract_metrics(
    definition: MetricsDefinition,
    trace: Trace,
) -> tuple[MetricRecord, ...]: ...
```

```python
class RecordStatus(StrEnum):
    MEASURED = "measured"
    NOT_APPLICABLE = "not_applicable"
    OPERATOR_FAILURE = "operator_failure"


MetricRecord = Annotated[
    MeasuredRecord | NotApplicableRecord | OperatorFailureRecord,
    Field(discriminator="status"),
]
```

```python
class EvalProcedure(FrozenModel):
    preprocessing: PreprocessingDefinition
    metrics: MetricsDefinition


class EvalPlan(FrozenModel):
    plan_id: str
    version: str
    task_set: TaskSet
    sampling_plan: SamplingPlan
    procedure: EvalProcedure
    aggregation: AggregationPolicy


def aggregate(request: AggregationInput) -> AggregationResult: ...
```

`evaluate_batch` runs one standalone bounded pool, while
`evaluate_durable_partition` runs cache misses serially without a nested pool.
Both stream terminal sample records through bundle-local shards or a supplied
`dr_store.ObjectStore`, retain only compact attempt membership and aggregate
state across the attempt, and optionally publish one terminal evaluation
bundle. Requested projections are written separately from authoritative
evidence and carry their source-attempt binding.

Evaluation bundles can be consumed at three grains:

- `read_eval_projection` verifies and validates only one fixed,
  self-bound projection artifact;
- `restore_eval_attempt` consumes the attempt and required record or
  reference shards without a preliminary whole-bundle audit; and
- `audit_eval_bundle` first verifies every artifact through dr-store,
  then validates the complete evaluation schema and reference graph without
  resolving external objects.

Object-store restoration resolves references sequentially under caller-owned
count limits. The released object store fully materializes and decodes each
canonical JSON record before dr-code can apply its strict record schema, so the
reader does not claim a caller-owned pre-decode byte or depth bound for those
external records.

`preflight_replay` reconstructs the complete ordered source attempt as frozen
samples or frozen materialized candidates. It returns `ReplayUnavailable`
without creating an attempt when recorded definitions or evidence are not
supported; `replay_eval_attempt` sends a ready replay through the same
standalone bounded batch path and records its source before publication.

`compare_eval_attempts` aligns compact attempt membership by slot and
sample identity. Equal references or content hashes remain compact; only
matched changed references are resolved, one pair at a time. Optional
projection definitions are explicit `(kind, left_version, right_version)`
tuples and yield either denominated comparable results or a typed
`ProjectionNotComparable` result.

`validate_preprocessing` and `validate_testing` are the standalone validation
flows over that machinery. Both run one caller-supplied request through
`evaluate_batch` and return its result, whose attempt carries the completeness,
validity, and limit-exhaustion verdicts. `validate_preprocessing` first runs the
request's distinct corpus texts through `preprocess_batch` under the plan's own
preprocessing definition and hands those traces to `evaluate_batch`, so the
corpus is preprocessed once and its `PreprocessingCoverage` — texts with
candidates, texts without candidates, texts whose preprocessing failed —
partitions the corpus under the definition actually evaluated. A reference
attempt plus an evidence resolver turns either flow structural, returning the
`compare_eval_attempts` result. The `dr-code-validate-preprocessing` and
`dr-code-validate-testing` verbs wrap those calls: they read one request
document, run the flow against a caller-named run root and a Python runtime
whose identity must match the request's, and print the verdict.

### [HumanEval+ evaluation](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/humaneval)

HumanEval owns the benchmark-specific task, evaluator job, and scoring policy.
Scoring is a projection over authoritative sample evaluation records; it never
starts a second candidate execution route. Results distinguish completed
benchmark outcomes from harness failure.

A scoring profile declares how the projection reduces a sample's candidates to
one outcome, and how it reduces the function groups within one candidate.
Extraction keeps one representation per candidate, so a solution and the helpers
written beside it share a candidate, and evaluation runs the complete suite once
per top-level function. `FIRST_CANDIDATE` scores candidate zero alone and
requires every one of its function groups to pass. `ANY_CANDIDATE_PASSES` scores
a pass when any candidate has any function group passing the complete suite, so
a failing helper cannot mask a correct solution; it reports a harness or
operator failure — never a measured zero — when no candidate passes but some
candidate's measurement is broken.

```python
class HumanEvalTask(FrozenModel):
    task_id: str
    prompt: str
    canonical_solution: str
    entry_point: str
    test: str
    ...


class CandidateReduction(StrEnum):
    FIRST_CANDIDATE = "first_candidate"
    ANY_CANDIDATE_PASSES = "any_candidate_passes"


class SubmissionOutcome(StrEnum):
    PASSED = "passed"
    TESTS_FAILED = "tests_failed"
    EVAL_INCOMPLETE = "evaluation_incomplete"
    EMPTY_SUBMISSION = "empty_submission"
    EXTRACTION_FAILED = "extraction_failed"
    NO_TOP_LEVEL_FUNCTIONS = "no_top_level_functions"
    TIMED_OUT = "timed_out"
```

```python
HumanEvalSubmissionResult = Annotated[
    CompletedSubmissionResult | HarnessFailure,
    Field(discriminator="kind"),
]


class HumanEvalSubmissionRequest(FrozenModel):
    sample: EvalSampleIdentity
    scoring_profile: HumanEvalScoringProfile


def project_humaneval_submissions_batch(
    records: Sequence[tuple[SampleEvalRecord, EvidenceReference]],
    requests: Sequence[HumanEvalSubmissionRequest],
) -> tuple[HumanEvalSubmissionResult, ...]: ...


def project_humaneval_submission(
    record: SampleEvalRecord,
    request: HumanEvalSubmissionRequest,
    *,
    sample_record: EvidenceReference,
) -> HumanEvalSubmissionResult: ...
```

### [Synthetic dataset generation](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/synthetic)

Synthetic datasets are built from versioned recipes whose corruption
components are deterministic for a source, settings model, and random state.
Each output carries the task, recipe, and seed that define its identity.
The identity also pins a digest of the exact ground-truth source being
corrupted. Dataset builds omit task/recipe pairs for which the recipe is not
applicable; direct single-sample builds report that case explicitly.

```python
class Recipe(FrozenModel):
    name: str
    version: str
    corruptions: tuple[CorruptionSpec, ...]
    description: str = ""


class Corruption(ABC, Generic[SettingsT]):
    NAME: ClassVar[CorruptionName]
    VERSION: ClassVar[str]
    Settings: ClassVar[type[CorruptionSettings]]

    @abstractmethod
    def apply(self, source: str, rng: random.Random) -> CorruptedSample: ...
```

```python
class SyntheticSample(FrozenModel):
    sample_id: str
    coordinate: SyntheticSampleCoordinate
    ground_truth_source: str
    corrupted_source: str


def build_dataset(
    tasks: Iterable[HumanEvalPlusTask] | None = None,
    recipes: Iterable[Recipe] = RECIPES,
    seed: int = 0,
    *,
    snapshot_path: Path | None = None,
) -> list[SyntheticSample]: ...
```

### [Code visualization](https://github.com/danielle-rothermel/dr-code/tree/main/viewer/packages/viewer)

The viewer package exposes domain-independent React primitives. Each accepts
plain content and semantic display options, leaving data loading and product
layout to its consumer.

```typescript
interface CodeBlockProps {
  code: string;
  lang?: string;
  theme?: "light" | "dark";
  className?: string;
}

interface CodeDiffProps {
  oldContent: string;
  newContent: string;
  oldName?: string;
  newName?: string;
  lang?: string;
  mode?: "split" | "unified";
  theme?: "light" | "dark";
}

interface StatusBadgeProps {
  status: "success" | "failure" | "warning" | "neutral";
  children: ReactNode;
  theme?: "light" | "dark";
  className?: string;
}
```

## Infrastructure

[`dr_code.core`](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/core)
contains the shared model, source, and execution foundations used across the
functional packages. It owns reusable mechanisms, while benchmark decisions
and measurement policy remain in their functional packages.

Candidate code executes through a pinned
[dr-exec](https://github.com/danielle-rothermel/dr-exec) executor:
`dr_code.evaluation.execution` builds one bounded importable-JSON job per
materialized candidate and interprets dr-exec's typed outcome and attribution
taxonomy into candidate, harness, and executor records. `dr_code.core.execution`
only provisions the caller-selected production executor. Submitted programs
are not contained by that process boundary: they retain the invoking worker's
permissions, and external worker isolation is the deployment boundary.

```python
class FrozenModel(BaseModel): ...


def host_process_executor(
    record_root: Path,
    *,
    runtime_executable: Path,
) -> ProcessExecutor: ...
```

## Development

Install the locked development environment and commit hook once per clone:

```console
uv sync --locked
uv run pre-commit install
```

The hook runs `scripts/pre-check.sh`, which verifies the locked environment,
Ruff formatting and lint, ty, `.defs`, the local Python suite, and the viewer
typecheck/build. Run `scripts/pre-check.sh --fix` explicitly when you want Ruff and ty to modify
the working tree.

The canonical local Python test run is serial:

```console
uv run pytest
```

For faster local feedback, run the same suite with an ephemeral xdist install;
CI remains serial so its ordering and resource use stay reproducible:

```console
uv run --with pytest-xdist pytest -n 4
```

Tests marked `postgres` need a live PostgreSQL-backed dr-store and are
deselected by default, so the command above stays offline. They cover the
evidence write path against a real database, where a fake cannot establish that
a rollback leaves nothing behind or that a first-writer-wins collision
surfaces. Set `DR_STORE_ROOT` to a dr-store checkout root and run:

```console
export DR_STORE_ROOT=/path/to/dr-store
scripts/run_postgres_tests.sh
```

The [viewer verification guide](viewer/README.md#verification) documents
typecheck, build, and gallery-based visual inspection.
