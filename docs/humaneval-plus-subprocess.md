# HumanEval+ host subprocess execution

HumanEval+ candidate evaluation launches a fresh copy of the active Python
interpreter for every request. The runner invokes `python -I -c <source>`,
writes bounded JSON to standard input, captures bounded output, and kills the
complete process group when the wall-clock deadline expires. It does not use or
require an OCI runtime or image.

The child receives a deliberately minimal environment rather than the parent
environment, and isolated Python mode ignores Python-specific environment
configuration. Provider credentials and other parent environment variables are
not inherited. The evaluation manifest records
`subprocess:python-isolated@v1` as its runner identity and retains the legacy
`sandbox_image` field as `null` for schema compatibility.

## Security boundary

A host subprocess is process separation, not a security sandbox. Isolated
Python mode and the minimal environment do not prevent candidate code from
reading files available to the current user, using the network, or starting
other processes. Run evaluations only on a disposable worker whose filesystem,
network access, credentials, and operating-system permissions are safe for
model-generated code. Do not run them on a developer workstation that contains
secrets or valuable uncommitted data.

## Preflight and tests

Sync the project environment, then verify that its isolated interpreter can
import NumPy. Some HumanEval+ tasks require it.

```sh
uv sync
uv run python -I -c \
  "import sys, numpy; print(sys.version.split()[0], numpy.__version__)"
```

The production candidate evaluator performs a stronger preflight before
creating worker threads: it imports NumPy and evaluates every pinned task's
canonical solution through the same subprocess runner used for candidates.
Run the focused runner and HumanEval checks locally with:

```sh
uv run pytest tests/humaneval
```

No runtime-specific environment variables or image preparation are needed.

## Corpus evaluation

Start a new evaluation directory when changing from an older OCI-backed run.
The runner identity is part of the execution coordinate, so existing OCI
SQLite state is intentionally incompatible with the host-subprocess backend.

```sh
uv run python scripts/evaluate_preprocessing_candidates.py \
  --preprocessing-run ../gen-viewer/data/preprocessing-runs/<run-id> \
  --corpus ../gen-viewer/data/generation-corpus.parquet \
  --output ../gen-viewer/data/candidate-evaluations/<new-run-id> \
  --snapshot tests/corpus/humanevalplus_snapshot.json \
  --max-workers 14
```

The command is resumable only when the source artifacts, pinned snapshot,
execution fingerprint, and subprocess runner identity still match.
