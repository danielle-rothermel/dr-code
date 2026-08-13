# Testing

This document describes the standard verification flow for dr-code. Follow it
for routine development, CI parity, and agent work.

## Do not run the baseline sweep unless asked

The directional HumanEval task-difficulty workflow under
`scripts/verification/task_difficulty/` is **not** part of the standard test
flow. It preprocesses and evaluates a large pinned generation corpus (production
defaults: ~203K nonblank generations, then a seeded sample of hundreds of
evaluations) and can take a long time.

**Do not run** `run_baseline.sh`, the numbered stage scripts (`01_`–`05_`), or
equivalent production-corpus commands unless the user explicitly requests a
baseline run or before/after comparison.

That workflow is a manual PR-versus-PR regression probe. Git-tracked exports
under `scripts/verification/task_difficulty/baseline/` are outputs from past
runs, not something CI or pytest produces.

For routine changes, the checks below are sufficient.

## One-time setup

```console
uv sync --locked
uv run pre-commit install
```

## Standard local flow

The commit hook and CI both converge on `scripts/pre-check.sh`. Run it before
pushing when you want full local parity:

```console
scripts/pre-check.sh
```

This runs, in order:

1. `uv sync --locked`
2. `uv run ruff format --check .`
3. `uv run ruff check .`
4. `uv run ty check`
5. `.defs` schema lint (`tombi` on `terms.toml` and `contracts.toml`)
6. `uv run pytest` (full Python suite, serial)
7. Viewer install, typecheck, and build (when `corepack` is available)

To apply autofixes for Ruff and ty, then rerun the checks:

```console
scripts/pre-check.sh --fix
```

Check output is cached under `.cache/pre-check/`.

## Python tests

Pytest covers `src/dr_code/` only. Repository scripts under `scripts/`, repo
metadata under `.defs/`, and the viewer workspace have no pytest coverage in
`tests/`. Scripts and the task-difficulty workflow are verified manually when
needed.

The canonical Python command is serial pytest:

```console
uv run pytest
```

Pytest is configured in `pyproject.toml`:

- Test root: `tests/`
- Import mode: `importlib`
- Tests marked `postgres` are **deselected by default** (`-m 'not postgres'`)

For faster local feedback, pytest-xdist is fine locally; CI stays serial:

```console
uv run --with pytest-xdist pytest -n 4
```

Run a focused subset while iterating:

```console
uv run pytest tests/evaluation/test_batch.py
uv run pytest tests/preprocessing/
```

### Opt-in PostgreSQL tests

Postgres-marked tests exercise the evidence write path against a real database.
They require a dr-store checkout and are not part of the default offline suite.

```console
export DR_STORE_ROOT=/path/to/dr-store
scripts/run_postgres_tests.sh
```

Pass additional pytest arguments to target specific files or tests.

## Viewer

The viewer workspace has no automated test suite. After viewer changes,
typecheck and build from `viewer/`, then verify behavior visually in the
gallery (`viewer/README.md#verification`).

```console
CI=1 corepack pnpm install --frozen-lockfile
corepack pnpm typecheck
corepack pnpm build
corepack pnpm --filter @dr-code/gallery dev
```

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on pushes to `main` and on pull
requests:

- **Python** (3.13 and 3.14): locked sync, ruff format/check, ty, `.defs` lint
  on 3.13 only, serial `uv run pytest`
- **Viewer**: frozen install, typecheck, build

Release tags additionally smoke-test the built wheel via
`scripts/check_built_wheel.py`.

## Manual verification (opt-in only)

These repository scripts are **outside the wheel** and are not invoked by
pre-check, pytest, or CI. Run them only when the task calls for them.

| Workflow | Entry point | Purpose |
|----------|-------------|---------|
| Task-difficulty baseline | `scripts/verification/task_difficulty/run_baseline.sh` | Full preprocess → sample → evaluate → summarize → export for before/after comparison |
| Task-difficulty fixture smoke | `DR_CODE_GENERATION_CORPUS_BUNDLE=tests/fixtures/generation_corpus/human_eval DR_CODE_TASK_DIFFICULTY_RUN_DIR=/tmp/task-difficulty-fixture scripts/verification/task_difficulty/run_baseline.sh fixture-smoke` | Small end-to-end smoke of the same stages |
| Preprocessing analysis | `scripts/analyze_preprocessing_success.py` | Preprocessing-only success rates on sampled tasks (`docs/verif_exps.md`) |
| Packaged validation CLIs | `dr-code-validate-preprocessing`, `dr-code-validate-testing` | Single-request validation flows through `evaluate_batch` |

## Contract checks

Individual contracts in `.defs/contracts.toml` name focused pytest commands for
specific invariants. The full suite already covers most of them; run a contract's
`check` command when you need targeted evidence for a contract you changed.
