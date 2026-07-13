# dr-code

Producer-blind HumanEval+ submission evaluator.

dr-code scores raw submission text against HumanEval+ tasks, parser profiles,
and scoring profiles. It owns the evaluator library and a localhost serve
facade for parser explanations; producers, orchestration, persistence, and
analysis live outside this repo.

## Ecosystem

Role: producer-blind evaluator for HumanEval+ task, submission, profile, outcome, metric, and explanation contracts.
Neighbors: dr-serialize, dr-providers, dr-graph, dr-platform, whetstone-ai, unitbench.
Consumers: whetstone-ai imports the library; unitbench consumes the serve facade.

## Surfaces

### Library API

The library surface is the curated `dr_code.humaneval` API plus four wide
general-purpose modules (ADRs 0006, 0007) usable directly from notebooks
and sibling repos:

- `code_transforms` / `code_analysis` operate on parseable Python (raise
  `SyntaxError` otherwise): transforms modify source or trees, analysis
  returns facts (annotation/docstring/signature sites, locals, equivalence).
- `text_transforms` / `text_analysis` are total, best-effort operations
  over text that probably contains code (fence splitting, code-likeness
  segmentation, cleanup).

The `dr_code.humaneval` modules:

- `code_parsing` extracts Python from submission text with versioned parser
  profiles.
- `task` models HumanEval+ tasks, parses test cases, and runs submissions in
  subprocesses.
- `scoring` combines extraction and task evaluation into score records.
- `metrics` builds text, Python-leakage, AST, compression, and task-test
  metrics.
- `profiles` resolves supported HumanEval scoring profiles.

The default scoring profile is `humaneval@v1`, using a 2.0 second sandbox
timeout and the `humaneval-best-effort@v1` parser profile. Unknown profile IDs
raise at the boundary.

### Generated-code sandbox

HumanEval candidates execute only in an OCI sandbox. The scorer passes bounded
JSON over stdin/stdout and exposes no host mounts or inherited application
environment. Each run has no network, a read-only root filesystem, a private
bounded `/tmp`, an unprivileged user, no Linux capabilities, no-new-privileges,
one PID, one CPU, and fixed memory, file-size, and open-file limits. Timeout
cleanup kills and removes the named container, which terminates its complete
cgroup rather than only the Python process.

Candidate-attributable terminations (memory/CPU-limit kills, interpreter
crashes, `SystemExit`, output floods) score as failed/error cases;
`HarnessFailure` is reserved for sandbox or runtime breakage so operators can
alert on it. The wall timeout includes container startup (roughly 0.1–0.5 s),
so keep configured timeouts comfortably above expected candidate runtime.
Candidate code shares the in-container interpreter with the trusted runner,
so the boundary guarantees host, credential, and network isolation — not
single-task score integrity against a deliberately adversarial submission.

Production requires Docker or Podman and the immutable image below to be
preloaded. Runtime image pulls are disabled, and scoring fails closed if the
runtime, daemon, exact digest, or required isolation flags are unavailable.

```bash
export DR_CODE_SANDBOX_RUNTIME=docker  # or podman
export DR_CODE_SANDBOX_IMAGE='python:3.13.14-slim@sha256:eb43ff125d8d58d7449dcba7d336c23bcac412f526d861db493b9994d8010280'
docker pull "$DR_CODE_SANDBOX_IMAGE"
```

The Whetstone `stores run` scoring worker must run on a host/VM with that OCI
runtime available (OrbStack/Docker on the current macOS operator path, or a
Linux Docker/rootless-Podman worker) and must preload the digest before queue
startup. Do not mount the runtime socket or any operator path into the sandbox.
Run scoring in its dedicated queue worker without provider credentials when
Whetstone queue selection is available; the sandbox still removes all
application credentials even when the trusted worker has them.

### Serve Facade

The optional serve facade exposes the parser explanation API over localhost:

```bash
uv run python -m dr_code.serve serve
uv run python -m dr_code.serve serve --port 8330
uv run python -m dr_code.serve openapi
```

Endpoints:

- `GET /health`
- `GET /profiles`
- `POST /explain`

The facade binds to `127.0.0.1`, allows localhost browser origins, and is meant
for local playgrounds and generated clients.

## Synthetic CLI

The synthetic CLI builds deterministic corruption datasets from the HumanEval+
snapshot:

```bash
uv run python -m dr_code.synthetic build \
  --recipes all \
  --tasks 10 \
  --seed 20260708 \
  --output /tmp/dr-code-synthetic.jsonl
```

Use `uv run python -m dr_code.synthetic build --help` for recipe selection and
output options.

## Setup

```bash
uv sync
uv sync --extra serve
```

The base install includes the evaluator library and synthetic CLI. The `serve`
extra adds FastAPI and uvicorn for the localhost facade.

## Tests

Run the full suite:

```bash
uv run pytest
```

Useful focused checks:

```bash
uv run pytest tests/humaneval
uv run pytest tests/synthetic
uv run ruff check .
```

Real denial probes are opt-in locally and always run when `CI` is set, so
they fail loudly in CI instead of silently skipping:

```bash
DR_CODE_RUN_SANDBOX_TESTS=1 uv run pytest tests/humaneval/test_sandbox.py
```

CI runs `uv sync`, `ruff check`, and `pytest` for pushes and pull requests.
