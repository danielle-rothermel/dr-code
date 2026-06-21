# Investigation Synthesis — Code Comp Pipeline Starting State

Recorded from repo inspection (June 2026). These notes describe **what exists** and **how pieces relate** — not a plan or roadmap.

Per-source detail:

- [dr-bottleneck](./dr-bottleneck.md)
- [nl-code](./nl-code.md)
- [code-eval](./code-eval.md)
- [dr-llm HumanEval pool extract](./dr-llm-humaneval-pool.md)
- [dr-providers](./dr-providers.md)

## Shared problem

Several repos orbit the same research question: **can an LLM compress code to natural language and reconstruct working Python?** They share `evalplus/humanevalplus` as a common dataset family but split the work across generation, extraction, and testing layers.

## Layer map

```text
┌──────────────────────────────────────────────────────────────────┐
│  GENERATION                                                        │
│  dr-providers · dr-bottleneck · nl-code (DSPy) · dr-llm (pools)   │
│  OpenRouter calls → raw text                                       │
└────────────────────────────┬─────────────────────────────────────┘
                             │ raw model output
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  EXTRACTION / PREP                                                 │
│  code-eval — LLMCodeValidator (extract, repair, validate, normalize)│
└────────────────────────────┬─────────────────────────────────────┘
                             │ extracted / normalized Python
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  TESTING                                                           │
│  nl-code — HumanEval+ test execution in Docker                     │
└──────────────────────────────────────────────────────────────────┘
```

Nothing wires these together today. Each repo is usable independently.

## Comparison at a glance

| | **dr-providers** | **dr-bottleneck** | **nl-code** | **code-eval** | **dr-llm pool** |
|---|------------------|-------------------|-------------|---------------|-----------------|
| **Primary job** | OpenRouter HTTP | Distributed enc/dec jobs | Benchmark load + test | Raw output validation | Historical decoder archive |
| **HumanEval tests** | — | Omitted from jobs | Parsed + executed | Not loaded | Not run (raw text only) |
| **Pass metric** | — | AST parse + compression | Test case pass rate | Validator pass | — (replay input) |
| **LLM calls** | Yes (core) | Yes (LiteLLM) | Yes (DSPy/LiteLLM) | No | Already done (203k rows) |
| **Maturity** | v0.1.0 library | Working pipeline | Production library | v0.1.0-frozen | One-off extract + artifacts |

## HumanEval handling compared

**Loading**:

- **nl-code**: richest — raw → derived `Task`, parsed tests, GT Docker verification, disk cache
- **dr-bottleneck**: flat HF dicts, docstring-stripped `source_code`, no tests on jobs
- **code-eval**: minimal `HumanEvalPlusTask` for synthetic corpus seeding only
- **dr-llm pool**: no loading — rows reference `human_eval_task_id`; prompts backfilled from nl-code cache

**Evaluation strictness** (weakest → strongest):

1. dr-bottleneck — `ast_parse_ok` + zstd size of encode output
2. code-eval — parseable Python + recovery provenance (no behavior check)
3. nl-code — functional HumanEval+ test cases in Docker

## Encoder-decoder experiments compared

**dr-bottleneck** and **nl-code** both run encode → decode workflows with character budgets and multiple models. Differences:

| Aspect | dr-bottleneck | nl-code |
|--------|---------------|---------|
| Orchestration | RabbitMQ multi-stage workers | In-process DSPy scripts |
| Encoder input | `{source_code}` from merged prompt+solution | `code_stub` or oracle `gt_code` |
| Decoder input | `{encode}` output | `function_stub` + generated spec |
| Scoring | AST + compression | Docker test pass rate |

**dr-llm pool data** is historical decoder output from dr-llm budgeted enc/dec pools (e.g. `budget_dec_v0_size6`). Same problem family, different infrastructure and template IDs.

## Two validation paths

### Replay (no new LLM calls)

Uses dr-llm pool extract as input:

```text
per_elem/*.parquet or *-dedup.jsonl
  → code-eval (validate raw_code_output)
  → nl-code (run_test_cases on human_eval_task_id)
```

Use Parquet when provenance matters (model, pool, template). Use dedup JSONL for dominant-pattern / extraction experiments.

This path can immediately quantify:

- How often real decoder output recovers to valid Python (code-eval)
- How often recovered code passes HumanEval+ tests (nl-code)
- Gap between dr-bottleneck-style AST pass and functional pass

### Generate fresh output

```text
nl-code tasks / prompts
  → dr-providers or dr-bottleneck (encode + decode)
  → code-eval
  → nl-code
```

dr-providers standardizes OpenRouter calls. dr-bottleneck adds scale (queues, lanes, budgets, repeats). nl-code remains the scoring authority.

## Transport duplication

Three OpenRouter call paths exist today:

| Repo | Mechanism |
|------|-----------|
| dr-providers | httpx, typed `LlmRequest` |
| dr-bottleneck | LiteLLM + Mongo logging |
| nl-code | DSPy + LiteLLM |

dr-providers is the thinnest and most explicit; the others add orchestration or framework integration on top.

## Synthetic vs empirical validation data

| Corpus | Source | Use |
|--------|--------|-----|
| code-eval synthetic | 4,100 controlled corruptions (164 × 25 recipes) | Attribution contracts, 99% recovery baseline |
| dr-llm pool dedup | 172,454 unique real decoder strings | Empirical distribution, dominant failure modes |

Together they answer different questions: code-eval asks "does the validator handle designed failures?"; pool data asks "does it generalize to production-like decoder mess?"

## Natural composition (conceptual)

No repo implements the full stack end-to-end. A coherent division of labor:

- **dr-providers** — how to call models
- **dr-bottleneck** — how to fan out enc/dec at scale
- **code-eval** — how to recover code from messy text
- **nl-code** — how to score correctness
- **dr-llm pool extract** — how to replay history without generation cost

dr-code (this repo) did not appear in the investigation scope; these notes live here as a central reference for cross-repo starting state.

## Key gaps (starting state, not action items)

- No shared schema linking `LlmResponse.text` → `ValidationResult` → `TestCaseResult` across repos
- dr-bottleneck evaluate step does not invoke code-eval or nl-code
- Pool replay and live generation use different provenance shapes
- nl-code already feeds dr-llm extraction (prompt backfill) but not code-eval or dr-bottleneck
