# code-eval — Investigation Notes

Sibling repo: `../code-eval`

Branch reviewed: `phase-3-normalization-and-freeze` (`v0.1.0-frozen`). Earlier main-branch Phase 1 stub is superseded.

## Purpose

Validation harness for **raw LLM Python output**: extract code from messy text, repair common failure modes, validate parseability, emit normalized forms and full provenance. Not a benchmark runner and not an LLM client.

## Datasets and task representation

HumanEval+ is used as **ground truth for a synthetic test corpus**, not as a live eval benchmark.

**Loading** (`synthetic/humaneval_loader.py`):

- `HumanEvalPlusTask`: `task_id`, `prompt`, `canonical_solution`, `entry_point`
- `full_source` = prompt + canonical_solution (no docstring stripping at load)
- HF download with offline snapshot fallback (`tests/corpus/humanevalplus_snapshot.json`)
- Test cases not loaded in current implementation

**Synthetic corpus** (`dataset_builder.py`, 25 corruption recipes, 22 inverse transforms):

- 164 tasks × 25 recipes → 4,100 `SyntheticSample` rows in `tests/corpus/synthetic_dataset.jsonl`
- Each sample: canonical ground truth, corrupted source, `expected_recovery_steps`
- Deterministic seeding from `(task_id, recipe_name, dataset_version)`

Corruptions simulate LLM failure modes: fences, prose wrappers, truncation, smart quotes, mangled imports, dead code, renamed locals, etc.

## Evaluation (what "eval" means here)

**Public API**: `LLMCodeValidator.validate(raw_output)` → `ValidationResult`

**Six-step pipeline**:

1. Capture raw input + fingerprint
2. Text normalize (NFC, CRLF→LF, tabs→spaces, trailing whitespace)
3. Extract — 8 extractors on raw and normalized input
4. Repair — per candidate: as-is, each repair alone, all repairs chained
5. Validate — AST parse, compile check, AST shape check (optional import resolution)
6. Normalize — 10 normalizers (L0–L5 + orthogonal forms)

**Success**: at least one candidate passes all validators (`overall_success`). Scientific output is **attributed recovery** (extractor path, repairs, normalizers).

**Property test results** (4,100-sample corpus):

- 99.0% overall recovery (4,061/4,100)
- Known hard cases: mid-function truncation recipes (~87%)
- Attribution checked against `expected_recovery_steps`

**Equivalence** (`equivalence.py`): syntactic via canonical AST round-trip — not runtime execution.

## Relation to other pieces

| Piece | Relationship |
|-------|----------------|
| **dr-bottleneck** | dr-bottleneck's `humaneval_compress_ast` is a single-step proxy (AST parse + zstd). code-eval is the full extraction/repair/normalization stack dr-bottleneck does not implement. |
| **nl-code** | Natural downstream consumer: `validate()` output → `evaluate_completed_code()` / `run_test_cases`. Separates "can we parse it?" from "does it pass tests?" |
| **dr-llm pool data** | Each `raw_code_output` is a direct `validate()` input. Real decoder mess (fences, prose, wrong names) complements the controlled synthetic corpus. Dedup JSONL useful for dominant-pattern sampling. |
| **dr-providers** | Strictly downstream: `response.text` from dr-providers feeds `validate()`. No dependency between the packages today. |

## Starting-state summary

- **Strength**: complete frozen validator, provenance logging, 99% recovery on synthetic corpus, subprocess-cached normalizers (ruff/ty)
- **Gap**: no HumanEval+ test execution, no LLM calls, no distributed orchestration
- **Natural role in a stack**: output sanitizer between generation and functional testing
