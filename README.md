# dr-code

## At a Glance

`dr-code` is a toolkit for preparing, evaluating, analyzing, and visualizing
Python code produced by language models.

- [Project website](https://danielle-rothermel.github.io/dr-code/)
- **Python package:** `dr-code` 0.1.0, for Python 3.13 and newer
- **React package:** `@dr-code/viewer` 0.1.0, for React 19
- **Danielle-owned repository dependencies:** None. The project currently
  depends only on third-party projects outside Danielle's GitHub account.

## High-level Design

The toolkit separates code evaluation into distinct, composable areas:

- **Candidate preparation** turns raw model responses into inspected Python
  candidates through declared, ordered preprocessing operations.
- **Trace capture** preserves intermediate artifacts, structured facts,
  failure reasons, and semantic provenance so results remain explainable and
  serializable.
- **Metric extraction** applies declared questions to traces and emits typed
  records for measurements, inapplicable questions, and operator failures.
- **Evaluation planning and scoring** identifies datasets, task selections,
  repeats, samples, and candidates, then reduces complete metric inputs into
  explicit score outcomes.
- **HumanEval+ evaluation** loads and samples benchmark tasks, extracts
  candidate solutions, runs them in an isolated Python sandbox, and reports
  structured outcomes.
- **Synthetic dataset generation** applies deterministic corruption recipes
  to known solutions for preprocessing and robustness experiments.
- **Code visualization** provides reusable React components for highlighted
  code, diffs, and status presentation, plus a private gallery for visual
  development.

The Python library owns the evaluation and data contracts. The React viewer
is a separate package in the same repository and does not ship in the Python
wheel.

## Development

Install the locked Python environment and run its checks:

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -q
```

The viewer workspace uses the Node version in `.nvmrc` and its own pnpm
lockfile:

```bash
cd viewer
pnpm install --frozen-lockfile
pnpm typecheck
pnpm build
pnpm test
```
