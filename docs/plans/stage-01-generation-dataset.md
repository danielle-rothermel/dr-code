# Stage 1 — Generation & dataset

[← Overview](./overview.md)

## Purpose

Produce a **unified dataset of decoder attempts**: each row describes what the decoder saw, what the model returned (raw, unparsed), and enough provenance to slice analysis later.

Two sources feed the same schema:

| Source | ID | Input |
|--------|-----|--------|
| **1a Pool replay** | `pool` | dr-llm HumanEval pool extract (Parquet / dedup JSONL) |
| **1b Fresh generation** | `fresh_stub` | HumanEval+ tasks + dr-providers batch decoder calls |

Future source (not v1): `fresh_encoded` — real encoder output at a budget (for DSPy and pool-comparable runs).

---

## Output shape: `AttemptRecord`

Canonical row consumed by stages 2–4. Exact field names TBD at implementation; logical shape:

```text
AttemptRecord
├── sample_id          # stable id for this eval unit (see dedup strategy)
├── run_id             # null for pool import; set for fresh batch runs
├── task_id            # e.g. "HumanEval/0"
├── entry_point        # function name under test
│
├── decoder_input      # text the decoder saw (description block content)
├── raw_output         # unparsed model text (maps to pool raw_code_output)
│
├── provenance
│   ├── source         # "pool" | "fresh_stub" | (later "fresh_encoded")
│   ├── model          # optional
│   ├── pool_name      # optional
│   ├── prompt_template_id / fingerprint  # optional
│   ├── enc_llm_config_id / dec_llm_config_id  # optional
│   ├── occurrence_count   # for deduped pool rows (default 1)
│   └── …              # preserve pool columns not listed here
│
└── task_ref           # optional embedded or lookup key to HumanEval+ tests
```

**Solidified rules:**

- `decoder_input` and `raw_output` are always non-empty strings for rows entering the eval pipeline.
- `source` is required; never merge pool and fresh_stub pass rates without slicing on it.
- Pool import maps existing columns (`human_eval_task_id`, `decoder_input_description`, `raw_code_output`, …) directly.
- Fresh runs populate the same fields; additional metadata goes under `provenance`.

---

## 1a — Pool loader

**Input artifacts** (external, documented in [dr-llm pool investigation](../investigation/dr-llm-humaneval-pool.md)):

- `humaneval_code_attempts.parquet` — full provenance
- `per_elem/human_eval-<n>-decode.parquet` — per-task
- `per_elem/human_eval-<n>-decode-dedup.jsonl` — `{ "out", "count" }` for dedup-first seeding

**Loader behavior:**

- Read Parquet for provenance-rich import, or JSONL dedup for eval seeding.
- Normalize `human_eval_task_id` → `task_id`.
- Map `decoder_input_description` → `decoder_input`, `raw_code_output` → `raw_output`.
- Set `provenance.source = "pool"`.
- For dedup JSONL: join back to Parquet when per-attempt metadata needed; otherwise carry `occurrence_count = count` and one representative `decoder_input` from a canonical row or task-default.

**Export formats:** Parquet and/or JSONL under a run directory (e.g. `exports/attempts/{dataset_id}.parquet`) for inspection before eval.

---

## 1b — HumanEval+ loader (dr-code owned)

**Decision:** Lightweight loader in dr-code — not nl-code’s full `Dataset` hierarchy.

**Minimum task model** (frozen once shipped):

```text
HumanEvalPlusTask
├── task_id
├── entry_point
├── prompt              # official stub (signature + docstring)
├── canonical_solution  # for GT reference only, not sent to decoder in v1
└── test                # raw test source string for stage 3 (or lazy load)
```

**Loading:**

- HuggingFace `evalplus/humanevalplus` test split, and/or offline snapshot (mirror code-eval pattern).
- Optional pinned revision for reproducibility.
- No ground-truth Docker verification at load time (nl-code does that if needed elsewhere).

**Tests:** Stage 3 needs parsed tests. Options (open question): store raw `test` string and parse in stage 3 via nl-code, or parse once at load time with shared logic.

---

## 1b — Fresh generation (dr-providers)

**Decoder prompt template** (aligned with dr-bottleneck `humaneval_encode_decode.yaml` decode step):

```text
Write functional code in Python according to the description.

"""
{description}
"""
```

**Description content (v1):** `task.prompt` — the official HumanEval function stub (signature + docstring). Not encoder-compressed text.

**Call path:**

```text
HumanEvalPlusTask.prompt  →  description
dr-providers OpenRouterProvider.generate(LlmRequest(...))
  →  raw_output = response.text
```

**Batch runner responsibilities:**

- Accept task filter (all / task ids / limit / seed sample).
- Model + sampling + reasoning via `LlmRequest` / `ReasoningSpec`.
- Rate limiting / concurrency (open question: in-process asyncio vs sequential vs dr-queues later for generation).
- Write `AttemptRecord` rows with `provenance.source = "fresh_stub"`.
- Record `run_id`, model, timestamps.

Generation does **not** need dr-queues in v1 unless batch volume requires it; eval pipeline does.

---

## Comparability notes

| Aspect | Pool (1a) | Fresh stub (1b) |
|--------|-----------|-----------------|
| Decoder template | description → code | Same template |
| Description text | Short encoder output | Full official stub |
| Difficulty | Harder | Easier (near-oracle) |
| Use | Realistic mess + provenance | Pipeline validation, decoder ceiling |

Both are valid; tag and slice separately in stage 4.

---

## CLI / scripts (target)

- `scripts/import_pool_attempts.py` — Parquet/JSONL → attempt export
- `scripts/generate_decoder_attempts.py` — HumanEval+ + dr-providers → attempt export
- Shared library: `dr_code.datasets.attempts`

---

## Solidified design points

- One schema (`AttemptRecord`) for all sources.
- dr-code owns minimal HumanEval+ loading.
- dr-providers for 1b LLM calls.
- Pool dedup JSONL is a first-class seed input for eval (with `occurrence_count`).
- Stub-as-description for v1 fresh runs; encoder output mode deferred.

---

## Open questions

- **`sample_id` strategy:** hash of `(task_id, raw_output)` vs pool `attempt_id` vs synthetic uuid for dedup rows?
- **Dedup JSONL without Parquet:** how to recover `decoder_input` when only `{out, count}` is present — task-level join, modal input from Parquet, or require Parquet for eval?
- **Snapshot pinning:** commit offline HumanEval+ snapshot in repo vs HF-only?
- **Test field handling:** raw string on task vs parse at load vs delegate entirely to nl-code in stage 3?
- **1b concurrency:** simple sequential batch vs parallel HTTP for generation phase?
- **Export location convention:** under repo `exports/` vs external data dir like pool artifacts?
