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

### [Preprocessing trace caching](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/caching)

`dr_code.caching` provides opt-in preprocessing trace memoization over a
[dr-store](https://github.com/danielle-rothermel/dr-store) record cache. It
accepts only validated entries whose input and producer match the request;
other cache outcomes fall through to fresh preprocessing. dr-store's managed
`SqliteRecordCache` supplies the persistent lifecycle.

While development mode keeps component versions at `"0"`, discard persistent
caches after preprocessing source, Python runtime, or dependency changes. Once
development mode ends, every such behavior-affecting change requires a version
bump for each affected preprocessing component before reusing its cache.

```python
def preprocessing_trace_cache_key(
    text: str,
    runner: BoundPreprocessingRunner,
) -> str: ...


async def run_preprocessing_cached(
    text: str,
    runner: BoundPreprocessingRunner,
    cache: RecordCache,
) -> Trace: ...
```

```python
from dr_store import SqliteRecordCache

async with await SqliteRecordCache.open("traces.sqlite3") as cache:
    trace = await run_preprocessing_cached(text, runner, cache)
```

### [Windowed execution caching](docs/windowed_execution_cache.md)

`WindowedExecutionCache` bulk-prefetches planned execution observations into a
bounded resident window. It retains at most one bounded persistence batch in
flight and one bounded pending batch; further outcomes remain memory-only.
Normal close attempts one final checkpoint. Persistent read, validation, and
write failures are logged and degrade to misses or dropped retry state rather
than failing evaluation.

Fresh execution observations become eligible for persistence only after the
owning sample evaluation record has a portable evidence reference. Batch
assembly publishes that reference with the observation, then releases the
resident cache key; reuse therefore never fabricates a source record or keeps
attempt-wide pending cache state.

Persistent keys combine the opaque execution-request key with the full digest
of a mandatory, caller-owned runtime identity. The identity must cover the
runtime, harness, dependency environment, and any other ambient behavior not
already represented by the request. The executor object itself is not part of
the persisted key.

```python
from dr_code.caching import WindowedExecutionCache
from dr_code.evaluation import EvaluationRuntimeIdentity
from dr_serialize import build_identity_document

runtime = EvaluationRuntimeIdentity(
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
class EvaluationProcedure(FrozenModel):
    preprocessing: PreprocessingDefinition
    metrics: MetricsDefinition


class EvaluationPlan(FrozenModel):
    plan_id: str
    version: str
    task_set: TaskSet
    repeat_plan: RepeatPlan
    procedure: EvaluationProcedure
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

- `read_evaluation_projection` verifies and validates only one fixed,
  self-bound projection artifact;
- `restore_evaluation_attempt` consumes the attempt and required record or
  reference shards without a preliminary whole-bundle audit; and
- `audit_evaluation_bundle` first verifies every artifact through dr-store,
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
supported; `replay_evaluation_attempt` sends a ready replay through the same
standalone bounded batch path and records its source before publication.

`compare_evaluation_attempts` aligns compact attempt membership by slot and
sample identity. Equal references or content hashes remain compact; only
matched changed references are resolved, one pair at a time. Optional
projection definitions are explicit `(kind, left_version, right_version)`
tuples and yield either denominated comparable results or a typed
`ProjectionNotComparable` result.

### [HumanEval+ evaluation](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/humaneval)

HumanEval owns the benchmark-specific task, evaluator job, and scoring policy.
Scoring is a projection over authoritative sample evaluation records; it never
starts a second candidate execution route. Results distinguish completed
benchmark outcomes from harness failure.

```python
class HumanEvalTask(FrozenModel):
    task_id: str
    prompt: str
    canonical_solution: str
    entry_point: str
    test: str
    ...


class SubmissionOutcome(StrEnum):
    PASSED = "passed"
    TESTS_FAILED = "tests_failed"
    EVALUATION_INCOMPLETE = "evaluation_incomplete"
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
    sample: EvaluationSampleIdentity
    scoring_profile: HumanEvalScoringProfile


def project_humaneval_submissions_batch(
    records: Sequence[tuple[SampleEvaluationRecord, EvidenceReference]],
    requests: Sequence[HumanEvalSubmissionRequest],
) -> tuple[HumanEvalSubmissionResult, ...]: ...


def project_humaneval_submission(
    record: SampleEvaluationRecord,
    request: HumanEvalSubmissionRequest,
    *,
    sample_record: EvidenceReference,
) -> HumanEvalSubmissionResult: ...
```

### [Synthetic dataset generation](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/synthetic)

Synthetic datasets are built from versioned recipes whose corruption
components are deterministic for a source, settings model, and random state.
Each output carries the task, recipe, and seed that define its identity.

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
permissions, external worker isolation is the deployment boundary, and
evaluations run only on disposable workers.

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
Ruff formatting and lint, ty, `.defs`, the local Python suite, and the viewer.
Run `scripts/pre-check.sh --fix` explicitly when you want Ruff and ty to modify
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

The [viewer verification guide](viewer/README.md#verification) documents its
independent typecheck, build, and test commands.
