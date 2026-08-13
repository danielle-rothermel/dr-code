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
uv sync --locked --group dev
uv sync --locked --group dev --group addons
uv run pre-commit install
```

## Standard local flow

The commit hook and CI both converge on `scripts/pre-check.sh`. Run it before
pushing when you want full local parity:

```console
scripts/pre-check.sh
```

This runs, in order:

1. `uv sync --locked --group dev`
2. `uv sync --locked --group dev --group addons`
3. `uv run ruff format --check .`
4. `uv run ruff check .`
5. `uv run ty check`
6. `.defs` schema lint (`tombi` on `terms.toml` and `contracts.toml`)
7. Core pytest with `DR_CODE_CORE_ISOLATION=1` (no `drc-*` wheels installed)
8. `scripts/run_addon_tests.sh`
9. Viewer install, typecheck, and build (when `corepack` is available)

To apply autofixes for Ruff and ty, then rerun the checks:

```console
scripts/pre-check.sh --fix
```

Check output is cached under `.cache/pre-check/`.

## Python tests

The repository is a **uv workspace**. Pytest covers `packages/*/tests/`.
The default **core** suite lives in `packages/dr-code/tests/` and runs with
only the `dr-code` wheel installed. Domain extensions are separate packages with
their own tests:

| Package | Tests | Runner |
|---------|-------|--------|
| `drc-humaneval` | `packages/drc-humaneval/tests/` | `scripts/run_humaneval_tests.sh` |
| `drc-synthetic` | `packages/drc-synthetic/tests/` | `scripts/run_synthetic_tests.sh` |
| `drc-generation-corpus` | `packages/drc-generation-corpus/tests/` | `scripts/run_generation_corpus_tests.sh` |

Run every extension test with:

```console
scripts/run_addon_tests.sh
```

The canonical **core** command:

```console
uv sync --locked --package dr-code --group dev
DR_CODE_CORE_ISOLATION=1 uv run --package dr-code --group dev pytest packages/dr-code/tests -m 'not postgres'
```

Core pytest is configured in `packages/dr-code/pyproject.toml` (import mode:
`importlib`; postgres tests remain opt-in).

For faster local feedback, pytest-xdist is fine locally; CI stays serial:

```console
uv run --with pytest-xdist pytest -n 4 packages/dr-code/tests
```

Run a focused core subset while iterating:

```console
uv run --package dr-code --group dev pytest packages/dr-code/tests/evaluation/test_batch.py
```

### Evaluation test helpers

Direct batch tests build requests with `request(..., preprocess_mode=...)`, which
controls whether sample preparation uses the bound runner (`in_process`) or the
dr-exec worker pool (`process_pool`).

`publish_batch()` in `packages/dr-code/tests/evaluation/_bundle_builders.py` is a bundle fixture
builder for reading, audit, replay, and publication tests. It defaults to
**frozen candidate inputs** and **no projections** for speed. Pass
`projections=(...)` when a test needs specific projection artifacts, and
`sample_inputs=True` when the published bundle must reflect live sample
preprocessing rather than frozen candidates.

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

- **Python core** (3.13 and 3.14): locked sync (`--group dev`), ruff, ty, `.defs`
  lint on 3.13 only, core pytest from `packages/dr-code/tests`
- **Python add-ons** (3.13 and 3.14): `uv sync --group dev --group addons` then
  `scripts/run_addon_tests.sh`
- **Viewer**: frozen install, typecheck, build

Release tags additionally smoke-test the built **`dr-code`** wheel via
`scripts/check_built_wheel.py`. Only `dr-code` is published to PyPI; addon
wheels are built in the workspace but not uploaded.

## Manual verification (opt-in only)

These repository scripts are **outside the wheel** and are not invoked by
pre-check, pytest, or CI. Run them only when the task calls for them.

| Workflow | Entry point | Purpose |
|----------|-------------|---------|
| Task-difficulty baseline | `scripts/verification/task_difficulty/run_baseline.sh` | Full preprocess → sample → evaluate → summarize → export for before/after comparison |
| Task-difficulty fixture smoke | `DR_CODE_GENERATION_CORPUS_BUNDLE=packages/drc-generation-corpus/tests/fixtures/generation_corpus/human_eval DR_CODE_TASK_DIFFICULTY_RUN_DIR=/tmp/task-difficulty-fixture scripts/verification/task_difficulty/run_baseline.sh fixture-smoke` | Small end-to-end smoke of the same stages |
| Preprocessing analysis | `scripts/analyze_preprocessing_success.py` | Preprocessing-only success rates on sampled tasks (`docs/verif_exps.md`) |
| Packaged validation CLIs | `dr-code-validate-preprocessing`, `dr-code-validate-testing` | Single-request validation flows through `evaluate_batch` |
| Addon CLIs (workspace only) | `drc-humaneval-schema`, `drc-synthetic` | HumanEval schema export and synthetic dataset CLI |

## Contract checks

Individual contracts in `.defs/contracts.toml` name focused pytest commands for
specific invariants. The full suite already covers most of them; run a contract's
`check` command when you need targeted evidence for a contract you changed.
