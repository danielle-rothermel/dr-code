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

Then produce deterministic compact Parquet tables, JSON, Markdown, and viewer
data with the evaluation manifest and its paired relations:

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
