# nl-code — Investigation Notes

Sibling repo: `../nl-code`

## Purpose

Research library for code-generation benchmarks: dataset loading, code execution (Docker-isolated), code analysis, and DSPy-based HumanEval experiments (direct vs encoder-decoder).

## Datasets and task representation

**Architecture** (`src/nl_code/datasets/`):

- Base `Dataset` class: load HuggingFace rows → parse into raw models → convert to derived `Task` objects → cache parsed snapshots to disk
- Flawed rows quarantined when parsing or ground-truth verification fails
- `DatasetSlice` for filtering, shuffle/seed, limits

**HumanEval** (`HumanEvalDataset`, `humaneval_task.py`):

- **Raw**: `RawHumanEvalTask` with nested `HumanEvalSource` (`prompt`, `canonical_solution`, `test`)
- **Derived**: generic `Task` (schema `v3`) with `target` (function name + kind) and `source.code` (runnable GT, docstrings/comments stripped)
- **Cached properties**: `gt_solution`, `test_suite` (parsed `HumanEvalTest`), `code_stub`, `function_stub`
- **Load-time validation**: ground-truth solutions verified in Docker via assertion tests

**Test parsing**:

- `HumanEvalTest` shapes: `inputs_results` (inputs + expected outputs) or `inputs_ref_func` (reference function)
- Parsed from the `check()` function body via AST helpers in `code_parsing.py`

**Scope**: HumanEval+, HumanEval-Pro, MBPP-Pro, BigCodeBench Lite Pro, ClassEval — each with its own raw model and dataset class.

## Evaluation

**Functional execution** (`code_execution/runner.py`):

- Modes: `function_call` (HumanEval), `assertion` (Pro benchmarks), `unittest` (ClassEval)
- Docker worker isolation via `dr-docker`; batch variants for throughput

**HumanEval DSPy eval** (`optim/humaneval_dspy_eval.py`):

1. Load `HumanEvalDataset().load()`
2. Generate via DSPy (direct or encoder-decoder)
3. `evaluate_completed_code()`: extract from fences → wrap with `run_single_test_case` helper → run per-case tests
4. `test_pass_rate` = fraction of cases passed; attempt passes at 100%

Skips tasks whose tests use `inputs_ref_func` (no expected outputs to compare).

**Enc/dec conventions** (`humaneval_dspy_sample.py`):

- `code_stub` — full prompt with docstrings (encoder input default)
- `function_stub` — docstrings stripped, comments preserved (decoder signature input)
- `gt_code` — oracle encoder input option

**Other eval primitives** (`evaluation/`): length/token counts and compression ratios for description-quality research (separate from functional pass/fail).

## LLM transport (current)

DSPy + LiteLLM via `configure_dspy_lm()` and OpenRouter catalog configs in `dspy_generators.py`. Not using the standalone `dr-providers` package.

## Relation to other pieces

| Piece | Relationship |
|-------|----------------|
| **dr-bottleneck** | Both run enc/dec HumanEval experiments; nl-code evaluates with real tests, dr-bottleneck with AST + compression proxies. nl-code's parsed HumanEval cache is already used by dr-llm pool extraction to backfill prompts. |
| **code-eval** | code-eval sits **upstream** of nl-code testing: recover parseable Python from messy LLM text before `run_test_cases`. nl-code's fence extraction is simpler than code-eval's full pipeline. |
| **dr-llm pool data** | Pool rows have `human_eval_task_id` joinable to nl-code's dataset; `raw_code_output` can be fed through code-eval then nl-code tests without re-running LLMs. |
| **dr-providers** | Could replace LiteLLM for non-DSPy batch scripts; DSPy eval path would remain separate unless refactored. |

## Starting-state summary

- **Strength**: typed datasets, parsed tests, Docker execution, functional pass rates
- **Gap**: not a distributed job runner; no RabbitMQ-scale orchestration like dr-bottleneck
- **Natural role in a stack**: benchmark truth layer — "does the code work?" after extraction/prep
