# dr-code

Dependency-clean nucleus for the compression–correctness question: given a
natural-language description of a HumanEval function, can a decoder
reconstruct working Python, and how compressible is the description?

The repo owns HumanEval+ parsing, scoring, and metrics as versioned profiles,
plus offline batch CLIs and a localhost serve facade. All dependencies are
public PyPI packages — a fresh clone builds with `uv sync` (the former
editable path deps on `../code-eval`, `../dr-providers`, and `../dr-queues`
were removed in the composable migration, PR #9).

## Layout

The wheel ships **two top-level packages** (`[tool.hatch.build.targets.wheel]`
in `pyproject.toml`) — don't install it alongside a standalone code-eval dist,
the `code_eval` package names collide:

- `src/dr_code/` — the nucleus:
  - `humaneval/` — parsing/scoring/metrics ported byte-identically from
    whetstone under existing profile IDs: scoring profile `humaneval@v1`
    (2.0s subprocess timeout), parser profiles `humaneval-best-effort` and
    `humaneval-field-marker` at `v1`, metrics profile `humaneval-metrics@v1`.
    `resolve_humaneval_scoring_profile` hard-raises on anything else.
  - `parsing/`, `testing/` — adapters over `code_eval` extraction/validation
    and HumanEval+ test execution for `AttemptRecord` batches.
  - `models/`, `datasets/`, `analysis/` — attempt/outcome schemas, pool and
    HumanEval+ snapshot loaders, offline join/aggregate/export.
  - `serve/` — FastAPI explain facade (requires the `serve` extra).
- `src/code_eval/` — code-eval 0.2.0 absorbed as in-repo source (extraction
  ladder, normalizers, repairs, validators, synthetic corpus tooling). Legacy
  lineage: excluded from `ty` checking pending the profile-v2 retyping.

## Setup

```bash
uv sync                # library + CLIs
uv sync --extra serve  # + FastAPI/uvicorn for the serve facade
```

## Serve facade

Localhost-only FastAPI app for the parser playground (default port 8321):

```bash
uv run python -m dr_code.serve serve            # run on 127.0.0.1:8321
uv run python -m dr_code.serve serve --port N   # other localhost port
uv run python -m dr_code.serve openapi          # dump OpenAPI schema
```

Endpoints: `GET /health`, `GET /profiles` (parser profile IDs + version),
`POST /explain` (stage-by-stage extraction explanation for one raw text).
CORS allows localhost origins only; the bind host is not configurable.

## Offline batch CLIs

```bash
uv run scripts/import_pool_attempts.py --help   # pool artifacts -> AttemptRecord exports
uv run scripts/parse_attempts.py --help         # AttemptRecord exports -> ParseOutcome JSONL
uv run scripts/test_attempts.py --help          # run HumanEval+ tests over parsed attempts
uv run scripts/analyze_eval_run.py --help       # join/aggregate exports for analysis
uv run scripts/build_humaneval_snapshot.py      # rebuild offline HumanEval+ snapshot (network)
```

Explore analysis outputs in marimo: `uv run marimo run nbs/analyze_eval_run.py`.

## Tests

Identity gates first — these pin the whetstone port to its golden fixtures
(doctrine: fix the port, never the fixture) and the v1 parser to the
4,100-sample corruption-corpus baseline:

```bash
uv run pytest -k "golden or corpus_baseline"
```

Full suite (nucleus units + ported code-eval suite + humaneval primitives):

```bash
uv run pytest
```

CI (`.github/workflows/ci.yml`) runs `uv sync`, `ruff check`, and the full
pytest suite on every PR. Scoring tests execute generated code in
subprocesses, so slow machines can surface timeout flakes.

## Historical docs

`docs/plans/` and `docs/adr/0001` describe the pre-migration Mongo/dr-queues
pipeline that PR #9 deleted; they are tagged **RETIRES AT D3 MERGE** and kept
only as history. Investigation notes live in `docs/investigation/`.
