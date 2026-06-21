# code-eval — work needed before prelim freeze

Review for using [code-eval](../code-eval) as a direct dependency in dr-code stage 2 (parsing). Recorded June 2026 after inspecting the `phase-3-normalization-and-freeze` branch.

Related: [Stage 2 plan](./plans/stage-02-parsing.md), [Investigation notes](./investigation/code-eval.md).

---

## Current status (2026-06-21)

**code-eval is preliminarily frozen for dr-code.** All must-fix and strongly-recommended items below landed on code-eval `main` at **`v0.1.1-frozen`** (package version `0.1.1`).

**dr-code dependency:** wired as an editable path dep (PyPI name conflict deferred):

```toml
# pyproject.toml
dependencies = ["code-eval==0.1.1", ...]

[tool.uv.sources]
code-eval = { path = "../code-eval", editable = true }
```

Keep `../code-eval` checked out at `v0.1.1-frozen` (or equivalent `main` with the same tree). `uv sync` verified — `import code_eval` resolves to `0.1.1`.

**Ruff override:** code-eval pins `ruff==0.8.4` for subprocess normalizers; dr-code dev lint uses `ruff>=0.15.18`. dr-code sets `[tool.uv] override-dependencies = ["ruff>=0.15.18"]`. Safe for stage 2 because parse workers should use `EXTRACTION_CONFIG` (`normalizers=()`), which skips normalization subprocess work entirely.

| Item | code-eval | dr-code |
|------|-----------|---------|
| 1. `EXTRACTION_CONFIG` | Done | Use in stage 2 parse workers (not wired yet) |
| 2. `best_valid_source()` / tie-break | Done | Use in stage 2 adapter (not wired yet) |
| 3. Public API test | Done | — |
| 4. Changelog + tag | Done (`v0.1.1-frozen`) | Path dep pins `0.1.1` |
| 5. Slim runtime deps | Done | Path dep avoids PyPI |
| 6. Pool-sample regression corpus | Done (39 samples) | — |
| 7. Documentation drift | Done | — |
| 8. Export enums / presets | Done (both exported) | — |
| 8. Path dep in dr-code | — | **Done** |
| Parse stage handler + `ParseOutcome` projection | — | **Not started** (stage 2) |

---

## Summary (original audit)

**You can use code-eval as a path dep today** — `LLMCodeValidator.validate()` → `ValidationResult` matches the dr-code plan. The packaging/API fixes listed below were required before calling it **preliminarily frozen**; they are now complete in code-eval `0.1.1`.

---

## What already works for dr-code

| Need | code-eval today |
|------|-----------------|
| Parse messy LLM text | Full extract → repair → validate pipeline |
| Success signal | `result.overall_success` |
| Code for nl-code stage 3 | `result.valid_candidates[i].source` |
| Provenance | `extractor_path`, `repairs_applied`, `extraction_log`, `normalizations` |
| Reproducibility | `config_fingerprint`, `tool_versions` |
| Pass-through metadata | `task_id` kwarg on `validate()` |
| Config customization | `ValidatorConfig` (validators, normalizers, cache_dir, timeouts) |
| Serialization | Pydantic v2 `model_dump(mode="json")` on `ValidationResult` |

Synthetic corpus performance (99% success, ~56s property suite with cache) is a solid gate for landing.

---

## Must-fix before prelim freeze (consumer contract)

> **Status:** all four items **done** in code-eval `0.1.1` / `v0.1.1-frozen`.

### 1. Named preset for extraction-only (performance) — done

**Issue:** `DEFAULT_CONFIG` runs **all 10 normalizers**, including six subprocess paths (ruff ×4, ty). Property tests averaged ~13ms/sample with cache; first run ~38ms/sample. At ~172k deduped pool rows, that is hours of parse-stage CPU/subprocess work **for normalization dr-code stage 3 does not need** (nl-code only needs `valid_candidates[].source`).

**Recommendation:** Add a documented, stable preset on the public surface, e.g.:

```python
EXTRACTION_CONFIG = ValidatorConfig(
    normalizers=(NormalizerName.L0_CANONICAL_AST,),  # or ()
)
```

Export it from `code_eval.__init__` alongside `DEFAULT_CONFIG`. Document in `USAGE.md` as **the recommended config for downstream test pipelines**. dr-code parse workers should use this, not `DEFAULT_CONFIG`.

This is the highest-impact change for the dr-code plan.

### 2. Document or implement best-candidate selection — done

**Issue:** `valid_candidates[0]` appears in `USAGE.md`, but order is **not** explicitly ranked/deduped despite `ARCHITECTURE.md` mentioning rank/dedupe steps that do not exist in code (`pipeline/` is only `steps.py` + `normalize_step.py`).

**Recommendation (pick one for freeze):**

- Add `ValidationResult.best_valid_source() -> str | None` (and optionally `best_valid_candidate()`), with a documented deterministic tie-break; **or**
- Export `Candidate` on the frozen surface and document tie-break rules explicitly.

Without this, dr-code will reimplement selection and the two repos may diverge silently.

### 3. Fix stale public API test — done

`tests/unit/test_public_api.py` still describes Phase 1 empty skeleton behavior. Update it to assert `overall_success` on clean input and non-empty `valid_candidates` — otherwise "freeze" is not guarded by CI.

### 4. Changelog + version tag discipline — done

`CHANGELOG.md` is still `[Unreleased]` for all of Phase 2–3. Before prelim freeze:

- Cut `[0.1.0]` or `[0.1.1]` release notes
- Tag e.g. `v0.1.0-prelim` or finalize `v0.1.0-frozen` after running the full suite
- dr-code path dep should pin that tag/commit

---

## Strongly recommended (low effort, high leverage)

> **Status:** all four items **done** in code-eval `0.1.1`.

### 5. Slim runtime dependencies — done

**Issue:** `datasets` and `typer` are **required** deps but only used by the synthetic harness / scripts — not by `validate()`.

**Recommendation:**

```text
core: pydantic, ruff (pinned)
optional [synthetic]: datasets
optional [cli]: typer
```

Keeps dr-code's dependency graph honest. `ruff` as a runtime dep is intentional (pinned normalizer behavior) — document that consumers need it available (it is vendored via the package).

### 6. Real-world regression corpus (small) — done

Synthetic 99% does not guarantee pool mess handling. Add ~20–50 lines to `tests/corpus/` sampled from `human_eval-0-decode-dedup.jsonl` (fenced, prose, wrong names) with tests that assert:

- `overall_success` rate (smoke threshold, not 100%)
- no crashes/exceptions on full `validate()`

This de-risks landing before dr-code hits 172k rows.

### 7. Fix documentation drift — done

- `ARCHITECTURE.md` references `extract_step.py`, dedupe, rank — does not match the tree
- `validator.py` module docstring still says "Phase 1 stub"
- Align docs with the actual pipeline so future agents do not plan against fiction

### 8. Export config enums or keep presets only — done

`USAGE.md` shows `from code_eval.names import NormalizerName` but `names` is documented as internal. For freeze, either:

- export `NormalizerName` / `ValidatorName` on the public surface, **or**
- forbid direct enum imports and expose only named configs (`DEFAULT_CONFIG`, `EXTRACTION_CONFIG`)

Mixed message today.

---

## Fine to leave in dr-code (no code-eval change required)

| Topic | Notes |
|-------|--------|
| **Slim `ParseOutcome` projection** | Full `ValidationResult` is large (`raw_input` duplicate + all candidates + normalizations). dr-code should project for Mongo, not store the whole result per row unless debugging. |
| **`task_id` unused internally** | Fine — metadata pass-through for the pipeline. |
| **Python 3.12 vs dr-code 3.13** | Compatible today; optional bump to `>=3.13` in code-eval for alignment. |
| **Known attribution gaps** | Issues #5–#8 (kitchen_sink, truncation, etc.) — synthetic corpus limitations, not blockers for pool replay. |
| **Import repair / wrong entry points** | Parser may succeed; nl-code tests fail — expected, not code-eval's job. |

---

## Optional later (post-prelim)

- `validate(..., normalize: bool = True)` shortcut — only if presets are not enough
- Stable JSON schema export for `ValidationResult` interop
- Publish to PyPI / git tag consumed by dr-code instead of a mutable path dep

---

## Suggested landing checklist

| Step | code-eval | dr-code |
|------|-----------|---------|
| 1. Full suite (`pytest` + slow + property) | Done (134 tests per CHANGELOG) | — |
| 2. `EXTRACTION_CONFIG` + export | Done | — |
| 3. Best-candidate helper + tie-break doc | Done | — |
| 4. Public API test + docstrings | Done | — |
| 5. Optional `datasets` / `typer` extras | Done | — |
| 6. Pool-sample regression corpus | Done | — |
| 7. Release notes + tag | Done (`v0.1.1-frozen`) | — |
| 8. Path dep + parse workers use `EXTRACTION_CONFIG` | — | Path dep **done**; parse workers **pending** (stage 2) |

---

## Bottom line

No architectural rework needed — calling `validate()` directly is correct. code-eval `0.1.1` closes the consumer-contract gaps (extraction preset, best candidate, dependency weight, CI/docs honesty, pool smoke corpus). dr-code has the path dependency; **next work is stage 2** — parse handler, `ParseOutcome` projection, and `EXTRACTION_CONFIG` + `best_valid_source()` at the call site.
