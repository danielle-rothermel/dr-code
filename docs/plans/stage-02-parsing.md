# Stage 2 — Parsing

[← Overview](./overview.md)

## Purpose

Turn each `AttemptRecord.raw_output` into a structured **parse outcome**: extracted Python (if recoverable), validator success/failure, and optional provenance from code-eval (extractors, repairs, normalizers).

Stage 2 is the **first queue stage** in the eval pipeline (stage 3 is Docker testing).

---

## Dependency

**code-eval** (`LLMCodeValidator`, `ValidationResult`) installed as a local path dependency on `../code-eval`. Call the public API; extend code-eval upstream when behavior gaps appear rather than forking logic into dr-code.

---

## Output shape: `ParseOutcome`

Projection of `ValidationResult` + links back to stage 1. Logical shape:

```text
ParseOutcome
├── sample_id, run_id, task_id    # join keys
├── raw_output                     # copy or hash reference
│
├── parse_success                  # ≥1 valid candidate (code-eval overall_success)
├── extracted_code                 # best valid candidate source (policy TBD)
├── candidate_count / valid_count
│
├── code_eval
│   ├── config_fingerprint
│   ├── extraction_log             # optional summary
│   ├── repairs_applied            # from best candidate
│   ├── extractor_path
│   └── normalizations             # optional subset (e.g. L0 canonical)
│
├── error / skip_reason            # if parse_success false
└── latency_ms                     # optional timing
```

**Best candidate policy (solidified direction):** prefer a valid candidate with the simplest extractor path, or first valid in deterministic order — document exact tie-break in implementation.

---

## Handler behavior

Per dr-queues job (one `AttemptRecord` or dedup batch unit in payload):

1. Read `raw_output` from job payload.
2. `validator = LLMCodeValidator()` (config from run manifest — open question).
3. `result = validator.validate(raw_output, task_id=task_id)`.
4. Project to `ParseOutcome`.
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

1. Unit tests: project `ValidationResult` → `ParseOutcome` on fixtures.
2. Golden samples: top dedup lines from `human_eval-0-decode-dedup.jsonl` (fenced, prose, wrong names).
3. code-eval synthetic corpus spot-check (optional cross-validation).
4. Integration: in-process `run_in_process` with a handful of jobs before detached workers.

---

## Solidified design points

- code-eval used directly; local path dep while in flux.
- Parse is queue stage 1 of eval (not a standalone script loop at scale).
- Preserve code-eval provenance fields useful for future granular analysis.
- Parse workers are in-process only (no Docker).

---

## Open questions

- **Forward on parse fail?** Always forward with skip flag vs terminal on fail (affects test queue volume).
- **ValidatorConfig:** per-run YAML vs `DEFAULT_CONFIG` only for v1?
- **Store full `ValidationResult` in Mongo** vs slim `ParseOutcome` only (storage size vs debuggability)?
- **Batch parse jobs:** one job per attempt vs mini-batch in one handler call?
- **Dedup jobs:** parse once per unique `raw_output`, fan out test with `occurrence_count` — implement in seeding or in handler?
