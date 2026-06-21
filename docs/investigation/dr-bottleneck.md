# dr-bottleneck — Investigation Notes

Sibling repo: `../dr-bottleneck`

## Purpose

Distributed RabbitMQ pipeline for running multi-stage LLM workflows at scale. The HumanEval-related work is an **encode → decode → evaluate** experiment: models compress code to a character budget, then reconstruct it.

## HumanEval loading and prep

**Loading** (`src/dr_queues/humaneval_data.py`):

- `load_humanevalplus()` fetches `evalplus/humanevalplus` test split via HuggingFace `datasets`
- Returns flat row dicts (`task_id`, `prompt`, `canonical_solution`, `entry_point`, etc.)
- Test cases are **not** carried on jobs (README notes they can be rejoined from the dataset later)

**Prepping**:

- `build_source_code()` concatenates prompt + canonical solution, then strips docstrings via AST
- `expand_experiment_jobs()` builds a cartesian product: lanes × tasks × budgets × repeats
- Each job is a `JobEnvelope` with `HumanEvalSampleInfo`, `HumanEvalJobMetadata` (budget), and `source_code`
- Default budgets: `[32, 64, 128, 256, 512, 1024]`
- `tiny_experiment_filters()` smoke mode: 2 tasks, one lane, budget 128

**Workflow** (`configs/workflows/humaneval_encode_decode.yaml`):

| Step | Kind | Role |
|------|------|------|
| `encode` | LLM | Summarize code within `{budget}` characters |
| `decode` | LLM | Reconstruct Python from encode output |
| `evaluate` | process | Handler `humaneval_compress_ast` |

Prompt context comes from `Workflow._prompt_context()` (`source_code`, `budget`, prior step outputs).

## Evaluation

Evaluation is **not** HumanEval+ functional testing. The process handler `humaneval_compress_ast` computes:

- `encoded_len_raw` — UTF-8 byte length of encode output
- `encoded_len_compressed` — zstd-compressed size (level 22)
- `ast_parse_ok` — 1 if decode output parses as valid Python AST, else 0

That AST check is the **`pass`** metric.

**Metrics export** (`metrics_report.py`): flat JSONL rows with model, budget, compression stats, and pass bit; `summarize_metrics()` groups by model and budget.

**Orchestration** (`scripts/run_humaneval_demo.py`): seeds RabbitMQ queues, runs workers in-process or detached, writes `exports/humaneval-{run_id}.jsonl` and `exports/metrics-{run_id}.jsonl`.

## LLM transport (current)

Uses `dr_bottleneck.llm.client.call_llm` (LiteLLM → OpenRouter) with Mongo call logging. YAML lane profiles map steps to model configs. This is separate from the standalone `dr-providers` package.

## Relation to other pieces

| Piece | Relationship |
|-------|----------------|
| **nl-code** | dr-bottleneck omits tests from jobs; nl-code has parsed test suites and Docker execution. dr-bottleneck's pass metric is a syntax proxy, not functional correctness. |
| **code-eval** | dr-bottleneck's evaluate step is a lightweight stand-in for what code-eval does richly on raw decoder text (extraction, repair, normalization). |
| **dr-llm pool data** | Historical decoder outputs from dr-llm pools overlap the decode step's output shape (`raw_code_output`-like text) but come from different pool/template configs. |
| **dr-providers** | dr-bottleneck could call OpenRouter through dr-providers instead of LiteLLM; orchestration and logging would stay in dr-bottleneck. |

## Starting-state summary

- **Strength**: scalable enc/dec sweep infrastructure (163+ tasks × models × budgets)
- **Gap**: no test execution, no output recovery beyond AST parse, no structured validation provenance
- **Natural role in a stack**: generation orchestration and throughput; downstream pieces handle extraction and testing
