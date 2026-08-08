# dr-code adversarial review ledger

**Date:** 2026-08-08

**Branch:** `08-08-dr-code-adversarial-review-fixes`

**Scope:** Python package, HumanEval evaluator, preprocessing, execution and metrics, persistence, synthetic data, analysis scripts, release workflow, and viewer package.

## Verdict

The review found ordinary inputs that could change correct code, misattribute failures, accept incompatible evidence, or report incomplete data as complete. The branch resolves every reproduced correctness finding selected by the review. Hardening, improvements, and style work remain documented below and are intentionally not implemented in this pass.

The review used independent read-only lanes for preprocessing/HumanEval, execution attribution, persistence/evaluation, synthetic data and scripts, viewer behavior, release architecture, and cross-cutting contracts. A finding entered the fix set only after a concrete reproduction against the branch.

## Correctness outcomes resolved on this branch

| Area | Current branch behavior |
| --- | --- |
| Import deduplication | Deduplicates only module-scope imports; equal imports in sibling lexical scopes remain intact. |
| Function discovery | Evaluates each distinct top-level function name once, so a redefinition cannot duplicate cached case results and create false incomplete coverage. |
| Import repair | Preserves valid parenthesized and backslash-continued imports, including an import sharing its closing line with another statement. |
| Missing-import inference | Uses lexical-scope-aware unresolved global loads; a parameter in one sibling function cannot suppress an import needed by another. |
| Source splitting | Recognizes an actual module-level `if __name__ == "__main__"` guard rather than matching strings, comments, or nested guards. |
| Synthetic source transforms | Preserves same-line statements when removing multiline imports and declines alpha-renaming in scopes that reflect names through `locals()` or `vars()`. |
| Candidate normalization | Extracts representations first, then normalizes each candidate only when the compiled AST remains equivalent. Unicode and multiline-string contents remain semantically intact. |
| JSON representation extraction | Ignores numeric token values while selecting the `code` field, so an enormous irrelevant number cannot abort candidate extraction. |
| Parsed HumanEval state | Parsed code, tests, cases, and nested values are immutable; derived parses are omitted from persistence and rederived from authoritative source/test text. |
| Check-function bindings | Expression-style checks bind the parsed argument name, including `check(fn)`, rather than assuming `candidate`. |
| Invalid candidate syntax | `CodeTest` records a full-coverage candidate error without issuing an execution request; it does not report an operator failure. |
| Trusted oracle failures | Trusted oracle evaluation is separated from candidate execution; an oracle exception invalidates the harness rather than becoming candidate evidence. |
| Trace identity | Setting equality distinguishes `bool`, `int`, and `float`, preventing wire-distinct producer coordinates from sharing cache identity. |
| Trace immutability | Serialized traces and JSON artifacts recursively freeze public nested containers while preserving dict/list wire shapes. |
| Producer coordinates | Persisted coordinates reject duplicate setting names, duplicate step instance names, and reserved `input`/`output` step names. |
| Aggregation identity | Aggregation rejects records with mismatched metric versions, metrics definitions, preprocessing producers, or slot provenance. |
| Aggregation units | Facts with the same name but incompatible units cannot be combined. |
| Synthetic identity | Sample coordinates, IDs, and RNG identity include a full digest of the exact ground-truth source. |
| Synthetic corruption strata | Fullwidth Unicode and quote-style recipes create actual applicable changes; non-clean no-op pairs are explicitly inapplicable and omitted from datasets. |
| Snapshot selection | Snapshot loading validates its pinned `test` split instead of ignoring a caller's requested split. |
| Sampling cardinality | Public sampling and the synthetic CLI fail when a requested count is non-positive or exceeds the available population. |
| Historical curve inputs | Malformed/non-finite measurements fail visibly instead of disappearing through null-skipping means. |
| Historical curve coverage | Every treatment is compared against one run-wide task/repeat population, so an entirely absent task is reported missing. |
| Historical curve labels | Figure titles use supplied model labels and neutral `Historical`/`Current` wording rather than hardcoded model/improvement claims. |
| Viewer grammars | `CodeDiff` requests exactly the five documented grammar families; unsupported `CodeBlock` languages stay plain text without an unhandled rejection. |
| Viewer declarations | The published declaration surface passes a strict dist-consumer check with `noUncheckedSideEffectImports`. |
| Release artifact gate | Linux builds the distributions; a supported macOS job downloads that exact artifact and executes the installed-wheel smoke path before publication. |

## Deferred hardening

These are real resilience gaps, but they strengthen guarantees beyond the correctness scope of this pass.

| Boundary | Gap | Recommended next action |
| --- | --- | --- |
| Candidate parsing | Extreme parser complexity can escape as `MemoryError`. | Define and enforce an explicit candidate parse-complexity/resource budget rather than broadly catching `MemoryError`. |
| Runner attribution | Process timeout, signal, output-limit, or kill outcomes do not identify whether trusted support/oracle code or candidate code owned the active phase. | Add an explicit phase-aware execution protocol before attributing resource exhaustion. |
| Runner result protocol | The atomic driver can return only all cases, yet a clean partial case list is accepted as candidate `evaluation_incomplete`. | Treat partial clean output as malformed runner/cache evidence for this driver version. |
| Execution cache liveness | Store reads/writes and writer `join()` are unbounded, so optional persistence can stall evaluation or close indefinitely. | Add bounded store-operation and shutdown policies at the persistence boundary. |
| Execution cache recovery | A `BaseException` from `put_many` can terminate the writer after the batch leaves dirty state; `close()` also does not coordinate an admitted blocked prefetch. | Preserve retry/in-flight state for terminal writer failures and define close/prefetch coordination. |
| Synthetic inputs | Duplicate task/recipe inputs and duplicate snapshot task IDs can create duplicate sample identities. | Reject duplicate identities at construction and snapshot-validation boundaries. |
| Artifact persistence | Synthetic JSONL and analysis outputs are written directly to final paths. | Serialize and validate to a temporary file, then atomically replace the destination. |
| Trace restoration | Restored value keys, step-fact owners, and `Absent` lineage are not cross-checked against declared producer steps. | Add cross-field referential validation to the persisted trace schema. |
| Persisted schema literals | Schema versions accept numerically equal non-integer values, and trace/cache payloads lack full golden compatibility fixtures. | Require exact integer schema-version types and add complete literal golden payloads. |
| Viewer async loading | Highlighter/dynamic-chunk rejections can remain unhandled and leave a permanent fallback; the singleton can retain a rejected promise. | Add explicit rejection state and controlled retry semantics. |
| Viewer package validation | The gallery aliases viewer source, so its green build does not prove packed-package exports/declarations work. | Add an installed/packed consumer smoke test independent of the source alias. |
| Viewer portability | Package build scripts use POSIX `rm -rf` without a supported-platform declaration. | Use a cross-platform clean command or explicitly scope supported developer platforms. |

## Deferred improvements

| Area | Opportunity |
| --- | --- |
| Evaluation-plan restoration | Reconcile the internally-complete plan contract with the intentional live-registry lookup in `tests/evaluation/test_plan.py`; choose either registry-free archived loading or wording that states the versioned runtime dependency. |
| Dependency footprint | Move Polars out of wheel runtime dependencies while only repository analysis scripts use it. |
| Cache integration evidence | Add a real `SqliteRecordCache` execution-cache close/reopen test. |
| Canonical serialization | Canonicalize mapping byte order if serialized traces become directly content-addressed. |
| Synthetic subset policy | Replace or explicitly name prefix selection for partial synthetic corpora; the current seed controls corruption, not task selection. |
| Viewer bundle footprint | Runtime grammar loading is constrained, but the vendor Shiki integration still exposes roughly 302 possible dynamic chunks in the gallery build. Replace that registry path if deployment size becomes important. |
| Viewer accessibility | Expose status semantics through accessible attributes and allow standard span/ARIA props. |
| Viewer package contents | Include license text and usage documentation in the packed artifact. |

## Deferred style

- Rename `test_environment_grant_is_fixed_and_hermetic`; it proves a fixed environment-variable grant, not hermetic containment.

## Validation record

Focused regressions accompany every resolved correctness item. The final branch gate is:

```bash
scripts/pre-check.sh
```

The viewer's package-specific checks are also exercised directly:

```bash
cd viewer
corepack pnpm typecheck
corepack pnpm test
corepack pnpm build
```

Implementation tip `e5dcb38c` passed the complete gate: 1,287 Python tests
passed with 5 expected xfails, all Ruff/ty/schema checks passed, and the
viewer passed typecheck, build, 34 tests, and its strict dist-consumer check.

## Review boundaries

- The result channel remains explicitly unauthenticated; this report does not claim candidate containment or forgery resistance.
- The evaluation set is HumanEval/HumanEvalPlus oriented; fixes preserve that intended scope.
- Deferred items are not release blockers for this branch unless a future claim adopts the stronger guarantee they describe.
