# dr-code — Pipeline overview

## Mission

**dr-code** is the research harness that standardizes evaluation for the compression–correctness question: given a natural-language description of a HumanEval function, can a decoder reconstruct **working** Python, and how compressible is the description?

It connects historical and fresh decoder outputs to:

1. **Parsing** — recover valid Python from messy LLM text
2. **Testing** — run HumanEval+ cases in Docker
3. **Analysis** — relate description compression (zstd) to test outcomes, sliced by experiment metadata

Ultimate downstream goal (not in initial scope): **DSPy optimization of encoder prompts** for the joint compression + correctness objective. dr-bottleneck will later adopt the same stage contracts for large-scale enc/dec runs.

Background: [Investigation synthesis](../investigation/synthesis.md).

---

## End-to-end flow

```text
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1 — Generation dataset                                     │
│  (a) dr-llm pool extract  OR  (b) HumanEval+ + dr-providers      │
│  → unified AttemptRecord rows                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stages 2–3 — Eval pipeline (dr-queues, v1)                       │
│  parse queue → code-eval workers (in-process)                      │
│  test queue  → Docker batch workers (nl-code)                      │
│  → MongoDB event / result store                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4 — Analysis                                               │
│  bulk script: zstd22(decoder_input) ⨝ test outcomes              │
│  marimo notebook: slices & charts                                │
└─────────────────────────────────────────────────────────────────┘
```

Stage docs:

- [Stage 1 — Generation & dataset](./stage-01-generation-dataset.md)
- [Stage 2 — Parsing](./stage-02-parsing.md)
- [Stage 3 — Testing](./stage-03-testing.md)
- [Stage 4 — Analysis](./stage-04-analysis.md)

---

## Major design decisions

### 1. Unified stage-1 schema for two sources

**Decision:** Pool replay (1a) and fresh generation (1b) must emit the same `AttemptRecord` shape so stages 2–4 are source-agnostic.

**Reason:** Avoid forked analysis paths; enable direct comparison once metadata tags distinguish sources.

### 2. Lightweight HumanEval+ loader in dr-code (not nl-code) for stage 1

**Decision:** Own a minimal, frozen HumanEval+ task model in dr-code (task id, entry point, official prompt stub, tests for stage 3).

**Reason:** nl-code’s dataset layer is rich but heavy for a single-benchmark harness; we only need a stable contract we control. nl-code remains the dependency for **execution**, not for stage-1 loading.

### 3. code-eval as a direct dependency (local path while in flux)

**Decision:** Stage 2 calls `LLMCodeValidator.validate()` from [code-eval](../../code-eval); install via path dependency `../code-eval` until published/stable.

**Reason:** Real pool outputs need full extract/repair/normalize provenance; reimplementing fences/repairs would drift. Update code-eval in place when needed.

### 4. dr-providers for stage 1b generation only

**Decision:** Fresh decoder runs use [dr-providers](../../dr-providers) (`OpenRouterProvider` + `LlmRequest`), not LiteLLM or DSPy.

**Reason:** Thin typed transport; DSPy comes later for encoder optimization, not for baseline batch collection.

### 5. Stub-as-description for initial fresh runs

**Decision:** For 1b, pass the HumanEval **function signature + prompt docstring** as the `{description}` inside the same decoder prompt template dr-bottleneck uses:

```text
Write functional code in Python according to the description.

"""
{description}
"""
```

**Reason:** Matches decoder **template shape** for pipeline validation. **Not** the same as pool rows (which use lossy encoder output at various budgets). Tag `provenance.source = pool | fresh_stub` in analysis.

**Later:** Add `fresh_encoded` mode (real encoder → description) before DSPy optimization.

### 6. dr-queues + MongoDB from v1 for stages 2–3

**Decision:** Implement parse and test as a two-stage [dr-queues](../../dr-queues) pipeline with MongoDB sinks from the first shipping version—not an in-process prototype replaced later.

**Reason:** Eval at pool scale (~172k deduped unique outputs × Docker tests) is the bottleneck; parallel multi-step eval is why the queue system exists.

**Shape:** Parse workers (cheap, in-process code-eval) → test workers (Docker batch via nl-code, spin up container per batch, tear down, next batch). Align `JobEnvelope` / handler registration with dr-queues conventions so dr-bottleneck can reuse later.

### 7. nl-code for stage 3 execution only

**Decision:** Do not reimplement Docker HumanEval+ testing.

**Reason:** nl-code already defines execution modes, batch runners, and infrastructure error contracts.

### 8. Dedup-aware job seeding

**Decision:** Prefer seeding eval jobs from **deduped** raw outputs (`out` + `count`) where possible; carry `occurrence_count` for weighted analysis.

**Reason:** Cuts Docker cost dramatically on pool replay without losing aggregate statistics.

### 9. Transport-agnostic result schemas

**Decision:** Define Pydantic models for `AttemptRecord`, `ParseOutcome`, and `TestOutcome` independent of RabbitMQ/Mongo layout.

**Reason:** Same schemas for unit tests, small local runs, and full queue pipeline; Mongo collections are projections.

### 10. Stage 4 reads Mongo (or export), not live queues

**Decision:** Analysis is offline against completed run artifacts.

**Reason:** Repeatable notebooks and scripts without coupling to runtime infrastructure.

---

## Local dependencies (initial)

| Package | Path / source | Used in |
|---------|---------------|---------|
| code-eval | `../code-eval` | Stage 2 |
| dr-providers | `../dr-providers` | Stage 1b |
| nl-code | `../nl-code` (TBD path dep) | Stage 3 |
| dr-queues | `../dr-queues` or published | Stages 2–3 orchestration |

---

## Repository layout (target)

Not prescriptive for phase planning, but intended direction:

```text
src/dr_code/
  datasets/          # HumanEval+ loader, pool loader, AttemptRecord
  generation/        # dr-providers batch runner, prompt templates
  parsing/           # code-eval adapter, ParseOutcome projection
  testing/           # nl-code adapter, TestOutcome projection
  pipeline/          # dr-queues workflow defs, handlers, seeding
  analysis/          # zstd joins, export helpers
scripts/             # typer CLIs per stage + full eval driver
nbs/                 # marimo analysis notebooks
docs/
  investigation/     # sibling repo notes (existing)
  plans/             # this directory
```

---

## Future steps (out of initial scope)

### dr-bottleneck integration

Replace dr-bottleneck’s AST-only evaluate step with calls into dr-code stages 2–4 (or shared libraries extracted from dr-code). Keep dr-bottleneck responsible for **LLM enc/dec orchestration at scale**; dr-code owns **eval semantics**.

Prerequisite: stable stage schemas and dr-queues handler modules importable from both repos.

### DSPy encoder optimization

Optimization loop:

```text
encoder prompt/program (DSPy)
  → encode (dr-providers or dr-bottleneck)
  → decode (fixed decoder prompt/template)
  → dr-code stages 2–4
  → scalar objective: f(zstd22(description), test_pass_rate, …)
```

Prerequisite: working eval pipeline, train/dev/eval task splits, and `fresh_encoded` generation mode (not stub-as-description only).

---

## Implementation phasing (suggested)

An agent picking up work should treat each bullet as a plannable phase; details live in stage docs.

1. **Schemas** — `AttemptRecord`, `ParseOutcome`, `TestOutcome`, run config
2. **Stage 1a** — pool Parquet/JSONL → `AttemptRecord` export
3. **Stage 1b** — HumanEval+ loader + dr-providers batch → same export
4. **Stage 2 handler** — code-eval adapter + unit tests on pool samples
5. **Stage 3 handler** — nl-code batch adapter + Docker smoke tests
6. **Pipeline** — dr-queues workflow (parse → test), Mongo sink, seed CLI
7. **Stage 4** — analysis script + marimo notebook on completed run
8. **Documentation** — update README, runbook for local RabbitMQ/Mongo

Cross-cutting: idempotent Mongo writes keyed by `(run_id, sample_id)`; parse-fail short-circuit to test stage with explicit skip reason.

---

## Open questions (cross-cutting)

See stage docs for stage-specific items. Repo-wide:

- **Mongo layout:** extend dr-queues `pipeline_events` only vs dedicated `eval_results` collection (or both)?
- **dr-queues dependency:** path dep on `../dr-queues` vs PyPI version pin?
- **nl-code dependency:** full package vs minimal execution import surface?
- **Test batch size default:** tune empirically (Docker startup vs throughput)?
- **Run manifest location:** align with dr-queues `.runs/{run_id}/manifest.json` convention?
