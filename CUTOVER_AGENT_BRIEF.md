# dr-code cutover — shared agent brief

TEMPORARY working file for the cutover build. Delete in the final wave.

You are migrating dr-code's HumanEval/eval machinery off its in-repo
subprocess primitive onto the **dr-exec** package. This brief is the
shared contract every wave depends on; your wave prompt adds the
wave-specific scope.

## Authoritative sources (read before touching code)

- dr-exec contract: `../dr-exec/docs/v1-design.md` (the executor API and
  its pinned semantics) and `../dr-exec/docs/dr-code-cutover.md` (the
  requirement-vs-artifact adjudication and the wave plan — your wave is
  described there). These are the law; where dr-code's old behavior
  conflicts, the adjudication says which wins.
- dr-exec API surface: `../dr-exec/src/dr_exec/__init__.py` exports the
  full public surface. Read `run.py`, `declare.py`, `record.py`,
  `batch.py`, `fake.py` for exact signatures. Do NOT modify dr-exec.

## The executor API you consume

- `run_tool(command, *, budgets, records, input_text="",
  environment=EnvironmentGrant.none(), exit_policy=REPORT_ONLY)`
- `run_untrusted_python(source, *, profile, budgets, records,
  runtime=HERMETIC, input_text="", environment=EnvironmentGrant.none(),
  exit_policy=REPORT_ONLY)`
- `run_untrusted_command(command, *, profile, budgets, records, ...)`
- `run_batch(request, *, profile, budgets, records, environment=...)`
- Outcomes are DATA: every spawned run returns a `RunResult` with a
  raw `returncode`, captured `stdout`/`stderr`, `truncation` marks,
  `measurements`, and exactly one `Outcome.attribution` (one of the
  `Attribution` StrEnum literals: payload, executor, channel, budget,
  machine, absence). A budget outcome names the `BudgetAxis`
  (wall_clock, output, input). NEVER dispatch on exception class for a
  run that spawned; branch on `result.outcome.attribution` and, for a
  batch, on per-item results.
- Exceptions are reserved for pre-spawn caller errors (`DeclarationError`)
  and executor failure where no result exists (`ExecutorFailure`).
- `EXECUTOR_IDENTITY` is the provenance string; `FakeExecutor` carries a
  distinct identity and refuses to claim the production one.
- `Records.directory(path)` or `Records.none()` — required per call.
- Budgets are caller-declared: `Budgets(wall_clock=..., output=...,
  input=...)` with `UNBUDGETED` where a bound is intentionally absent.
- Env grants: `EnvironmentGrant.none()/named(vars)/fixed(mapping)/
  overlay(extra, exclusions)` — construction-time-frozen, introspectable
  (`.declared_names`, `.contents_digest()`).
- Testing: consumers use `FakeExecutor` (behavioral scripting) for logic
  tests; parity/oracle tests and driver-body tests that must genuinely
  execute a payload run the REAL engine with `Records.none()` (sanctioned
  by the contract). Consumers never test spawn-path *correctness*.

## Standing conventions (repo owner's rules — non-negotiable)

- Hard cutover: DELETE the old path, do not deprecate or leave compat
  shims. Old and new never coexist. Approved removals: remove the thing
  and every reference in one pass.
- Docs/docstrings/comments are forward-facing: never reference the old
  primitive, "ported from", or removed things. Sweep prose
  (README, docs/) in the same pass as code.
- No magic strings: persisted/wire literals live in StrEnums with
  uniqueness verification and golden tests; never derive wire keys from
  field names. Payload-observed exception identity (what the CANDIDATE
  raised) is payload data — preserved byte-for-byte; only EXECUTOR
  failure vocabulary changes to dr-exec attribution literals.
- Frozen slotted dataclasses for internal value objects; pydantic
  BaseModel only at serialization boundaries. Nested composition over
  flattening.
- Recorded dev data is archived, never destroyed; but no recorded data
  justifies preserving a schema.

## Dependency mechanics (established in wave 01, relied on after)

- dr-exec is a local path dependency: `pyproject.toml` gets
  `[tool.uv.sources]` `dr-exec = { path = "../dr-exec", editable = true }`
  and `dr-exec` in `[project].dependencies`. After adding, run
  `uv sync` (NOT `--locked` — the lock must regenerate).
- CI (`.github/workflows/ci.yml`) must check out dr-exec as a sibling so
  `uv sync` resolves on runners: add a second `actions/checkout@v7` step
  with `repository: danielle-rothermel/dr-exec`, `path: dr-exec`, and
  `ref` (a validation item — Danielle confirms the cross-repo checkout
  credential). Because the runner workspace root holds the dr-code
  checkout, use a uv source path that matches CI's layout; if the
  natural relative path differs between local (`../dr-exec`) and CI,
  prefer a layout that works for both (document the choice).
- Packaging test (`tests/packaging/test_installed_viewer_wheel.py`):
  a path source does not survive into wheel metadata and dr-exec is not
  yet on an index, so its clean-install and byte-reproducibility
  assertions cannot pass while the path source is in force. Mark them
  `xfail(reason=...)` naming the pin-swap commit as the re-enable gate.
  These land in the wave that carries that test (see the wave plan).

## Git/Graphite (non-interactive only)

- Your branch is stacked on the CURRENT branch. Create it with
  `gt create -am "<single-line message>" <exact-branch-name>`; add
  commits with `gt modify -ca -m "<message>"`. NEVER git commit/branch/
  rebase/push; NEVER gt submit (the orchestrator submits).
- Before starting: `git -C /Users/daniellerothermel/drotherm/repos/dr-code status -s` (expect clean apart from this brief file, which is committed in wave 01 and deleted in the final wave) and `git branch --show-current` (confirm your expected base).

## Verification (bounded — do exactly this)

- `uv sync` (regenerate lock after any dependency change), then the
  quality gate the repo's CI runs: `uv run ruff format --check .`,
  `uv run ruff check .`, `uv run ty check`, `uv run pytest`. Python
  suite should complete in a few minutes; if any command exceeds ~5
  minutes, abort it and report. Do NOT run the viewer pnpm/playwright
  chain (out of scope for execution waves; note if a change would
  affect it). Fix failures before committing.
- Your final message is a report: what you changed, files touched,
  which dr-code behaviors changed and how they map to the accepted-
  behavior-changes list, the test count/result, any place the
  adjudication was ambiguous and the reading you chose, and the branch
  you created.
