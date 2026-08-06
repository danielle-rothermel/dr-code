# dr-code

[![CI](https://github.com/danielle-rothermel/dr-code/actions/workflows/ci.yml/badge.svg)](https://github.com/danielle-rothermel/dr-code/actions/workflows/ci.yml)

[Terms and contracts](https://danielle-rothermel.github.io/dr-code/) ·
[terms source](https://github.com/danielle-rothermel/dr-code/blob/main/.defs/terms.toml) ·
[contracts source](https://github.com/danielle-rothermel/dr-code/blob/main/.defs/contracts.toml)

**Personally owned dependencies:** [dr-store](https://github.com/danielle-rothermel/dr-store).

**dr-code prepares, evaluates, analyzes, and visualizes Python code produced by
language models.**
The repository contains a Python library and a separately packaged React
viewer, organized into these functional areas:

- **[Candidate preparation](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/preprocessing)**
  turns raw model responses into inspected Python candidates through declared,
  ordered preprocessing operations.
- **[Trace capture](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/trace)**
  preserves intermediate artifacts, structured facts, failure reasons, and
  semantic provenance so results remain explainable and serializable.
- **[Measurement](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/metrics)
  and [evaluation](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/evaluation)**
  extracts typed measurements from traces, declares evaluation plans, and
  reduces complete measurement slots into typed aggregation outcomes.
- **[HumanEval+ evaluation](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/humaneval)**
  loads and samples benchmark tasks, extracts candidate solutions, runs them
  in an isolated Python sandbox, and reports structured outcomes.
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
    provides the shared isolated-execution boundary.
  - **[Preprocessing trace caching](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/caching)**
    memoizes preprocessing traces through a caller-supplied record cache.

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


def extract_metrics(
    definition: MetricsDefinition,
    trace: Trace,
    *,
    run_in_sandbox: SandboxRunner = ...,
    execution_cache: ExecutionCache | None = None,
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

### [HumanEval+ evaluation](https://github.com/danielle-rothermel/dr-code/tree/main/src/dr_code/humaneval)

HumanEval owns the benchmark-specific task, extraction, sandbox protocol, and
scoring policy. Scoring returns a discriminated result so a completed scoring
outcome cannot be confused with harness failure.

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
HumanEvalSubmissionScore = Annotated[
    CompletedScore | HarnessFailure,
    Field(discriminator="kind"),
]


def score_humaneval_submission(
    *,
    raw_submission: str,
    task: HumanEvalTask,
    scoring_profile_id: str = ...,
    scoring_profile_version: str = ...,
    run_in_sandbox: SandboxRunner = ...,
) -> HumanEvalSubmissionScore: ...
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

```python
class FrozenModel(BaseModel): ...


class SandboxRunner(Protocol):
    def __call__(
        self,
        *,
        source: str,
        input_json: str,
        timeout_seconds: float,
    ) -> SandboxCompletedProcess: ...
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


def run_preprocessing_cached(
    text: str,
    runner: BoundPreprocessingRunner,
    cache: RecordCache,
) -> Trace: ...
```

```python
from dr_store import SqliteRecordCache

with SqliteRecordCache("traces.sqlite3") as cache:
    trace = run_preprocessing_cached(text, runner, cache)
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

The canonical local Python test run is serial and excludes the live Docker
probes:

```console
uv run pytest -m "not oci"
```

For faster local feedback, run the same suite with an ephemeral xdist install;
CI remains serial so its ordering and resource use stay reproducible:

```console
uv run --with pytest-xdist pytest -n 4 -m "not oci"
```

Tests marked `oci` require Docker and the digest-pinned sandbox image. They
skip locally unless `DR_CODE_RUN_SANDBOX_TESTS=1`; CI runs them separately.
The [viewer verification guide](viewer/README.md#verification) documents its
independent typecheck, build, and test commands.
