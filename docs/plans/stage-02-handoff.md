# Stage 2 handoff — Parsing

Brief context for the agent implementing Stage 2. Read the full design in [stage-02-parsing.md](./stage-02-parsing.md) and [overview.md](./overview.md) first.

---

## Your mission

Build the **code-eval adapter** that turns each `AttemptRecord.raw_output` into a `ParseOutcome`, then wire it as the first handler in the dr-queues eval pipeline. Stage 2 scope is the parsing logic + unit tests; full queue/Mongo orchestration is stage 2+3 together (see phasing in overview), but the adapter must be queue-ready.

---

## What Stage 1 left you

### Input contract: `AttemptRecord`

Defined in `src/dr_code/models/attempts.py`. Key fields for parsing:

| Field | Notes |
|-------|-------|
| `sample_id` | SHA-256(`task_id` + `\0` + `raw_output`)[:16] — stable dedup key |
| `run_id` | `None` for pool import; set for fresh runs |
| `task_id` | e.g. `"HumanEval/0"` — pass to `validator.validate(..., task_id=...)` |
| `entry_point` | function name under test |
| `raw_output` | **Unparsed model text** — often fenced (`` ```python ... ``` ``), sometimes bare code |
| `decoder_input` | Not used in stage 2; needed later for zstd analysis |
| `provenance.source` | `"pool"` or `"fresh_stub"` — slice analysis on this, never merge blindly |
| `provenance.occurrence_count` | >1 for deduped pool rows; carry through for weighted stats |

### Ready-made exports to parse against

After `uv run scripts/demo_stage1.py`:

- `exports/demo/pool.parquet` — 7 fixture rows (mixed sources, fenced outputs)
- `exports/demo/fresh.parquet` — 1 live fresh_stub row

Load with `dr_code.datasets.export.read_attempts(path)`.

Pool fixtures also live at `tests/fixtures/pool/`. code-eval golden samples: `../code-eval/tests/corpus/pool_samples.jsonl` (39 real dr-llm outputs).

### Skeleton you extend

`src/dr_code/models/outcomes.py` has a minimal `ParseOutcome` — expand it to match [stage-02-parsing.md](./stage-02-parsing.md) (code_eval provenance, candidate counts, etc.).

`ParseOutcome`/`TestOutcome` are intentionally transport-agnostic; Mongo layout comes later.

---

## Dependencies already wired

| Dep | Status | Stage 2 usage |
|-----|--------|---------------|
| `code-eval==0.1.1` | Editable `../code-eval` | `LLMCodeValidator`, `EXTRACTION_CONFIG`, `ValidationResult` |
| `dr-providers` | Editable `../dr-providers` | Not used in stage 2 |
| `dr-queues` | **Not wired yet** | Needed for pipeline orchestration (stage 2–3) |

**Critical:** use `EXTRACTION_CONFIG` (extract → repair → validate, `normalizers=()`). Do **not** use `DEFAULT_CONFIG` at pool scale — it runs subprocess ruff/ty normalizers.

**Critical:** use `result.best_valid_source()` / `best_valid_candidate()` for extracted code. Do not reimplement tie-breaking or pick `valid_candidates[0]`.

Integration tracker: [code-eval work needed](../code-eval-work-needed.md).

---

## Suggested implementation shape

```text
src/dr_code/parsing/
  adapter.py      # validate(raw_output, task_id) → ParseOutcome
  config.py       # EXTRACTION_CONFIG re-export / fingerprint helper
```

1. **`parse_attempt(record: AttemptRecord) -> ParseOutcome`** — thin wrapper around code-eval.
2. **Unit tests** on `pool_samples.jsonl` + rows from `exports/demo/pool.parquet` / fixtures.
3. **Optional local CLI** — `scripts/parse_attempts.py` reading an export Parquet and writing parse results (useful before dr-queues is wired).

Queue handler (later, with dr-queues): read `AttemptRecord` from job payload → call adapter → write `step_outputs["parse"]` → forward to test stage.

---

## Things we learned in Stage 1

1. **`raw_output` is messy.** Live Gemini output came back fenced with full stub echoed inside; pool fixtures mix fenced and bare code. code-eval is the right layer — don't regex in dr-code.

2. **Pool vs fresh_stub `decoder_input` differs by design.** Pool rows may have short encoder text or full stub (fixtures include both); fresh_stub always uses `task.prompt`. Parsing only cares about `raw_output`.

3. **Dedup semantics.** Same `raw_output` on one task → same `sample_id`. Parse once per unique `(task_id, raw_output)` at scale; propagate `occurrence_count` to analysis, not necessarily re-parse.

4. **Export round-trip works.** Parquet flattens provenance to columns + JSON extra blob; JSONL uses full nested Pydantic dump. Either format is fine for seeding parse jobs.

5. **HumanEval+ snapshot is offline.** 164 tasks at `tests/corpus/humanevalplus_snapshot.json`. Stage 3 will need `HumanEvalPlusTask.test` (raw string); stage 2 only needs `task_id`.

6. **OpenRouter profiles** live in `configs/openrouter_profiles.yaml` — irrelevant to parsing, but fresh exports record `provenance.dec_llm_config_id`.

---

## Verification targets

```bash
uv run pytest tests/unit -q
# Parse adapter + golden pool_samples:
uv run pytest tests/unit/test_parsing_adapter.py -q
# Single-example walkthrough:
uv run scripts/demo_stage2.py --show-failure
# Batch parse export:
uv run scripts/parse_attempts.py --input exports/demo/pool.parquet --output exports/demo/parse.jsonl
```

Success criteria for stage 2 adapter (before queue):

- Pool fixture fenced outputs → `parse_success=True` with plausible `extracted_code`
- Known-bad raw text → `parse_success=False` with `skip_reason`
- Provenance fields populated from `best_valid_candidate()` when successful
- No subprocess normalizers in default path (`EXTRACTION_CONFIG`)

---

## Open decisions (resolve during stage 2)

From [stage-02-parsing.md](./stage-02-parsing.md):

- Forward to test stage on parse fail vs short-circuit?
- Store full `ValidationResult` in Mongo vs slim `ParseOutcome`?
- Dedup: parse once in seeding vs handler?

Recommend: **forward with skip flag** (keeps pipeline shape uniform); **slim ParseOutcome in Mongo** with optional debug export; **parse once per sample_id** when seeding from dedup exports.

---

## Docs & plans

- Design: [stage-02-parsing.md](./stage-02-parsing.md)
- Stage 1 (complete): [stage-01-generation-dataset.md](./stage-01-generation-dataset.md)
- code-eval investigation: [../investigation/code-eval.md](../investigation/code-eval.md)
