# dr-llm HumanEval Pool Extract — Investigation Notes

Source doc: `../dr-llm/docs/humaneval-pool-extraction.md`

Artifact run: `/Users/daniellerothermel/drotherm/data/code-comp/dr-llm-humaneval-pool-dumps/20260621_manual` (outside repos; not committed)

## Purpose

One-off extraction of HumanEval-related **decoder code-generation attempts** from historical `dr-llm` pool projects. Builds a broad analysis dataset for parsing, clustering, and validating extraction/testing pipelines on **real** model output — without re-running LLMs.

## What was extracted

**Scope**:

- Pool projects: `code_comp_t1`, `code_comp_v0`, plus `nl_latents` (audited; zero HumanEval rows under exact policy)
- Exact HumanEval ID policy: `human_eval/HumanEval/<n>` only; excludes HumanEvalPro and other datasets
- Decoder/direct code attempts only (`response_json.text`; not standalone encoder rows)

**Scale** (completed run):

- 203,407 extracted attempts
- 163 per-task Parquet files
- 172,454 task-local unique raw strings (deduplicated)
- 26 dumped pools, ~1.4 GB total artifact dir

**Description backfill**: uses nl-code parsed HumanEval cache when pool payload lacks prompt text (`humaneval_cache.prompt` — 9,856 rows).

## Artifact shapes

| File | Contents |
|------|----------|
| `humaneval_code_attempts.parquet` | Unified table (~102 MB) |
| `per_elem/human_eval-<n>-decode.parquet` | Same columns, one task |
| `per_elem/human_eval-<n>-decode-dedup.jsonl` | `{"out": "<raw>", "count": N}`, sorted by repeat count |

**Key columns**:

- `raw_code_output` — unparsed model text (fences, prose, wrong names preserved)
- `decoder_input_description` — NL spec the decoder saw
- `human_eval_task_id` — e.g. `HumanEval/0`
- Provenance: `pool_name`, `model`, `dec_llm_config_id`, prompt template IDs, encoder lineage, timestamps

Dedup JSONL is for quick inspection; Parquet retains full provenance per attempt.

## Observed raw output patterns

Sampled from `human_eval-0`, `human_eval-20`, `human_eval-100` dedup files (~960–1,260 unique strings per task):

- Markdown fences (` ```python ... ``` `) — often highest repeat counts
- Prose wrappers before fences ("Here's a functional Python code snippet…")
- Wrong entry points (correct-ish logic, wrong function name)
- Extra scaffolding (imports, `main()`, stdin parsing)
- Clean minimal code without fences
- Verbose preambles with wrong implementations

Overlaps code-eval failure modes but reflects **empirical** decoder output from budgeted enc/dec pools (dominant pool: `budget_dec_v0_size6`, ~146k rows).

## Relation to other pieces

| Piece | Relationship |
|-------|----------------|
| **code-eval** | Each `raw_code_output` → `LLMCodeValidator.validate()`. Measures recovery/attribution on real generations vs synthetic 4,100-sample corpus. Dedup files prioritize high-`count` dominant modes. |
| **nl-code** | Join on `human_eval_task_id` → `HumanEvalDataset` → functional tests on extracted code. Quantifies parse-only vs actually-works gap (e.g. wrong-name outputs that AST-parse but fail tests). |
| **dr-bottleneck** | Same high-level enc/dec experiment family; pool data is historical decoder output from dr-llm pool configs, not dr-bottleneck's RabbitMQ workflow. Comparable problem shape, different provenance. |
| **dr-providers** | Bypassed for replay — data already contains raw outputs. New generation runs could populate future pool-like tables via dr-providers. |
| **nl-code (again)** | Extraction scripts already depend on nl-code's HumanEval cache for prompt backfill. |

## Starting-state summary

- **Strength**: 203k real decoder attempts with provenance; 172k unique raw strings; per-task splits for targeted analysis; marimo notebook for exact dedupe inspection
- **Gap**: not a maintained library API (scripts under `../dr-llm/scripts/`); artifacts live outside repo; dedup JSONL drops per-attempt provenance
- **Natural role in a stack**: offline replay corpus for validating extraction (code-eval) and testing (nl-code) without LLM cost
