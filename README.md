# dr-code

[![CI](https://github.com/danielle-rothermel/dr-code/actions/workflows/ci.yml/badge.svg)](https://github.com/danielle-rothermel/dr-code/actions/workflows/ci.yml)

| [Project website](https://danielle-rothermel.github.io/dr-code/) |
| --- |

**dr-code prepares, evaluates, analyzes, and visualizes Python code produced by
language models.**
The repository contains a Python library and a separately packaged React
viewer, organized into these functional areas:

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
