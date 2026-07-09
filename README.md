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

The default scoring profile is `humaneval@v1`, using a 2.0 second subprocess
timeout and the `humaneval-best-effort@v1` parser profile. Unknown profile IDs
raise at the boundary.

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

CI runs `uv sync`, `ruff check`, and `pytest` for pushes and pull requests.
