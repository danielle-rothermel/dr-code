# dr-code

## Preprocessing boundary

Preprocessing extracts structurally usable code from text. A definition may
require each returned code candidate to contain a top-level function, but it
does not require that function to have an application- or benchmark-specific
name. Function identity is intentionally outside the repository's generic
preprocessing contract.

Applications that need a particular function name can add an explicit
processing step or filter preprocessing or test results afterward. Executable
tests, rather than name matching in preprocessing, determine whether an
extracted candidate satisfies a task.

The preprocessing trace is the authoritative record of semantic analysis. It
owns stable terminal failure codes, extraction provenance, per-stage candidate
counts, rejection reasons, compilation diagnostics, and structural function
facts. Batch and persistence adapters may attach source/run identity and
mechanically reshape those facts, but should not reclassify preprocessing
results. A missing external value remains an ingestion concern because there
is no text artifact to process.

The public HumanEval flow is
`humaneval-function-candidates@v1`. Bind it once for batch work; every
successful output is a nonempty ordered candidate set whose entries compile,
contain at least one top-level function, and carry stable candidate IDs plus
their extraction origins. `[[ ## code ## ]]` is supported as an input
representation inside this flow, not as a separate parser mode.

```python
from dr_code.preprocessing import (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
    bind_preprocessing,
)

runner = bind_preprocessing(
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION
)
```

See the [decoder-output preprocessing analysis plan](docs/decoder-output-preprocessing-plan.html)
for the flow’s design and the reproducible corpus audit built on it.

For full HumanEval+ scoring that requires NumPy, see the
[reproducible sandbox-image build and preflight flow](docs/humaneval-plus-sandbox.md).

## Corpus evaluation and analysis

Completed preprocessing runs contain `results.parquet`, `candidates.parquet`,
`step_facts.parquet`, and `rejections.parquet` plus a complete manifest. The
candidate evaluator preserves every `(sample_id, candidate_id, candidate_index)`
membership while deduplicating execution by task and source. Its
`candidate_membership.parquet` relation joins those sample candidates to
`candidate_results.parquet` through `evaluation_key`; analysis always validates
the full join before computing test rates.

The repository-adjacent shared-artifact convention is
`../gen-viewer/data/preprocessing-runs/<run-id>/` for preprocessing and
`../gen-viewer/data/candidate-evaluations/<run-id>/` for test results. These
full artifacts are intentionally not committed. Run or resume evaluation from
an explicit source checkout and pinned HumanEval+ snapshot:

```bash
DR_CODE_SANDBOX_IMAGE='sha256:<locally-built-image-id>' \
  uv run python scripts/evaluate_preprocessing_candidates.py \
  --preprocessing-run ../gen-viewer/data/preprocessing-runs/<run-id> \
  --corpus ../gen-viewer/data/generation-corpus.parquet \
  --output ../gen-viewer/data/candidate-evaluations/<run-id> \
  --snapshot tests/corpus/humanevalplus_snapshot.json \
  --max-workers 14
```

Then produce deterministic compact Parquet tables, JSON, Markdown, and a
`viewer-data.json` analysis snapshot with the evaluation manifest and its paired
relations:

```bash
uv run python scripts/analyze_preprocessing_corpus.py \
  --corpus ../gen-viewer/data/generation-corpus.parquet \
  --run-dir ../gen-viewer/data/preprocessing-runs/<run-id> \
  --candidate-membership ../gen-viewer/data/candidate-evaluations/<run-id>/candidate_membership.parquet \
  --candidate-results ../gen-viewer/data/candidate-evaluations/<run-id>/candidate_results.parquet \
  --candidate-evaluation-manifest ../gen-viewer/data/candidate-evaluations/<run-id>/candidate_evaluation_manifest.json \
  --output-dir analysis/preprocessing/<run-id>
```

Both CLIs fail closed on incompatible or partial artifacts. The analysis API
is also available as `dr_code.corpus.analyze_preprocessing_corpus`.

Before rescoring a newly generated preprocessing run, export an append-only
identity audit against the immutable baseline:

```bash
uv run python scripts/compare_preprocessing_runs.py \
  --corpus ../gen-viewer/data/generation-corpus.parquet \
  --before-run ../gen-viewer/data/preprocessing-runs/<before-run-id> \
  --after-run ../gen-viewer/data/preprocessing-runs/<after-run-id> \
  --output-dir analysis/preprocessing-comparisons/<before-run-id>--<after-run-id>
```

The comparison writes sample outcome transitions, candidate membership/source
changes, and normalized provenance-path deltas as deterministic Parquet rows,
plus a reconciled JSON summary and immutable-input manifest. To compare
existing evaluation artifacts too, supply both `--before-evaluation` and
`--after-evaluation`; supplying only one is rejected. Existing output paths are
never overwritten.

## Local corpus viewer

The interactive viewer is a local FastAPI application backed by DuckDB. It
queries registered corpus, preprocessing, and optional candidate-evaluation
Parquets dynamically; it does not use the analyzer's `viewer-data.json`
snapshot as its data source.

Build the React frontend before starting the Python service:

```bash
cd viewer
pnpm install --frozen-lockfile
pnpm --filter @dr-code/preprocessing-analysis build
cd ..
```

Each run is registered with a JSON descriptor. This canonical descriptor uses
paths relative to the descriptor file, although absolute paths are also
accepted:

```json
{
  "label": "baseline",
  "corpus": "../data/generation-corpus.parquet",
  "preprocessing": "../data/preprocessing-runs/<run-id>",
  "candidate_evaluation": "../data/candidate-evaluations/<run-id>"
}
```

The descriptor must be a JSON object, and its accepted fields are exact:
`label` is optional, but must be a string when no external label is supplied;
exactly one string path named `corpus` or `corpus_path` is required; exactly one
string path named `preprocessing`, `preprocessing_manifest`, or
`preprocessing_manifest_path` is required; and at most one string path named
`candidate_evaluation`, `candidate_evaluation_manifest`, or
`candidate_evaluation_manifest_path` may be supplied. The candidate evaluation
is optional. A preprocessing or candidate-evaluation path may identify either
its artifact directory or its manifest file. Unknown fields and multiple
aliases for the same path are rejected. The CLI's `LABEL=` prefix supplies the
external label and overrides `label` in the descriptor.

Start one or more named runs and choose where the persistent DuckDB database
lives:

```bash
uv run dr-code viewer \
  --run label=descriptor.json \
  --database .runs/dr-code-viewer.duckdb
```

Repeat `--run LABEL=descriptor.json` to load additional runs.

Open `http://127.0.0.1:8000`. This unauthenticated development tool binds only
to loopback addresses (or `localhost`) and rejects non-loopback `--host`
values. The Waterfall view traces stage counts to examples, Compare shows
compatible run deltas and terminal transitions, and Review groups terminal
failures for verdicts, notes, and tags.

Review annotations persist in the selected DuckDB database across restarts and
can be downloaded as deterministic JSON from `GET /api/annotations/export`.
The database stores run provenance and annotations, not copies of the source
relations: all registered Parquet files remain external and must stay available
at their descriptor-resolved paths while the viewer is running.
