# Stage 2 — Parsing

[← Overview](./overview.md)

**Status:** Done (2026-06-21). Adapter in `src/dr_code/parsing/`, expanded `ParseOutcome`, unit tests, `scripts/parse_attempts.py`, and `scripts/demo_stage2.py`. dr-queues handler wiring deferred to stage 2–3 pipeline phase. Entry doc for implementers: see completed work in repo; stage 3 handoff is [stage-03-handoff.md](./stage-03-handoff.md).

## Purpose

Turn each `AttemptRecord.raw_output` into a structured **parse outcome**: extracted Python (if recoverable), validator success/failure, and optional provenance from code-eval (extractors, repairs).

Stage 2 is the **first queue stage** in the eval pipeline (stage 3 is Docker testing).

---

## Dependency

**code-eval** `0.1.1` (`LLMCodeValidator`, `ValidationResult`, `EXTRACTION_CONFIG`) — editable path dependency on `../code-eval`, already wired in dr-code `pyproject.toml`. PyPI publish is deferred (name conflict).

Call the public API; extend code-eval upstream when behavior gaps appear rather than forking logic into dr-code. Integration status: [code-eval work needed](../code-eval-work-needed.md).

**Parse-stage config (v1 default):** `EXTRACTION_CONFIG` — extract → repair → validate only (`normalizers=()`). Do **not** use `DEFAULT_CONFIG` at pool scale (~172k rows); it runs all 10 normalizers including subprocess ruff/ty.

---

## Output shape: `ParseOutcome`

Projection of `ValidationResult` + links back to stage 1. Logical shape:

```text
ParseOutcome
├── sample_id, run_id, task_id    # join keys
├── raw_output                     # copy or hash reference
│
├── parse_success                  # result.overall_success
├── extracted_code                 # result.best_valid_source() (None if parse failed)
├── candidate_count / valid_count
│
├── code_eval
│   ├── config_fingerprint
│   ├── extraction_log             # optional summary
│   ├── repairs_applied            # from best_valid_candidate()
│   ├── extractor_path             # from best_valid_candidate()
│   └── normalizations             # empty with EXTRACTION_CONFIG; omit or debug-only
│
├── error / skip_reason            # if parse_success false
└── latency_ms                     # optional timing
```

**Best candidate (solidified):** use `ValidationResult.best_valid_source()` / `best_valid_candidate()`. Tie-break (ascending = preferred), defined in code-eval:

1. Fewest `repairs_applied`
2. Shortest `extractor_path`
3. Raw extraction over text-normalized duplicate
4. Lexicographic `candidate_id`

Do not reimplement selection in dr-code.

---

## Handler behavior

Per dr-queues job (one `AttemptRecord` or dedup batch unit in payload):

1. Read `raw_output` and `task_id` from job payload.
2. `validator = LLMCodeValidator(config=EXTRACTION_CONFIG)` (per-run YAML override optional later).
3. `result = validator.validate(raw_output, task_id=task_id)`.
4. Project to `ParseOutcome`:
   - `parse_success = result.overall_success`
   - `extracted_code = result.best_valid_source()`
   - provenance from `result.best_valid_candidate()` when present
5. Write outcome to job `step_outputs["parse"]` / `step_records["parse"]`.
6. Emit pipeline event (Mongo via dr-queues sink).
7. Forward job to test stage **always** (test stage records skip if parse failed) **or** short-circuit with explicit skip — see open question.

**Worker type:** in-process Python worker — no Docker. CPU-bound; scale horizontally with worker pool size.

---

## Queue integration

**Pipeline position:** stage 1 export → **seed parse queue** → parse workers → test queue.

Uses [dr-queues](../../dr-queues):

- `JobEnvelope.payload` carries serialized `AttemptRecord` (or reference + fields needed downstream).
- Handler registered in `HandlerRegistry` (module path via `--handlers-module` for detached workers).
- Events appended to MongoDB before ack/forward (dr-queues invariant).

Parse stage is cheap relative to test stage; still queue-backed from v1 so the full pipeline shape is real on day one.

---

## Failure modes

| Case | Handling |
|------|----------|
| Empty raw output | Reject at stage 1 import; if seen, `parse_success=false` |
| code-eval throws | Retry policy via dr-queues/worker; record infrastructure error |
| No valid candidate | `parse_success=false`, no `extracted_code`; test stage skips Docker |

Distinguish **parse failure** (model/output problem) from **infrastructure failure** (retry).

---

## Testing strategy (for implementers)

1. Unit tests: project `ValidationResult` → `ParseOutcome` on fixtures (`EXTRACTION_CONFIG`).
2. Golden samples: start from [code-eval `tests/corpus/pool_samples.jsonl`](../../code-eval/tests/corpus/pool_samples.jsonl) (39 real dr-llm decoder outputs); extend with top dedup lines from `human_eval-0-decode-dedup.jsonl` as needed.
3. code-eval synthetic corpus spot-check (optional cross-validation).
4. Integration: in-process `run_in_process` with a handful of jobs before detached workers.

---

## Solidified design points

- code-eval used directly; path dep wired at `0.1.1` / `v0.1.1-frozen`.
- Parse workers use **`EXTRACTION_CONFIG`**, not `DEFAULT_CONFIG`.
- Extracted code via **`best_valid_source()`**; do not use `valid_candidates[0]`.
- Parse is queue stage 1 of eval (not a standalone script loop at scale).
- Preserve code-eval provenance fields useful for future granular analysis (repairs, extractor path).
- Parse workers are in-process only (no Docker).

---

## Open questions

- **Forward on parse fail?** Always forward with skip flag vs terminal on fail (affects test queue volume).
- **Per-run ValidatorConfig override?** v1 uses `EXTRACTION_CONFIG` only; YAML/manifest override deferred. Use `DEFAULT_CONFIG` only for debug/research runs.
- **Store full `ValidationResult` in Mongo** vs slim `ParseOutcome` only (storage size vs debuggability)?
- **Batch parse jobs:** one job per attempt vs mini-batch in one handler call?
- **Dedup jobs:** parse once per unique `raw_output`, fan out test with `occurrence_count` — implement in seeding or in handler?
