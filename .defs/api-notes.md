# API naming proposals — dr-code

Proposals surfaced while maintaining `.defs/vocab.html`. **Proposals only — do not implement from a doc pass.** Any rename touching definition identities, hashes, run-id labels, goldens, or fixtures is fixture-regenerating and must not change values in a doc task.

## Package surface: no top-level `__all__`

- **Current:** `src/dr_code/__init__.py` is a deliberate empty namespace (8 lines, no `__all__`). The public surface lives in per-subpackage `__all__` lists: `eval` (51 names), `preprocessing`, `trace`, `metrics`, `humaneval`, `synthetic`, `corpus`, `viewer`.
- **Problem:** The vocab doc's colophon claim "Exported names match `src/<pkg>/__init__.py`" cannot be satisfied at the package root — there is no single package-level public surface to key an Exported Names table to. The doc scopes to one subpackage (`dr_code.eval`, the evaluation kernel) and says so.
- **Proposal:** Decide and record the convention — either (a) each subpackage keeps its own vocab scope, or (b) a documented union surface. If the kernel remains the contract surface, keep `dr_code.eval` as the doc's target and leave the root namespace as-is.
- **Tooling note (flag to skill maintainer, not a doc edit):** `check_exports.py` globs `src/*/__init__.py` and takes the FIRST match — here `src/dr_code/__init__.py`, which has no `__all__` — so it FAILS to verify a repo whose public surface is a sub-package. Manual comparison against `src/dr_code/eval/__init__.py` confirms all 51 `__all__` names appear exactly once in the names column, no missing, no extras, no dupes. Fix options: (a) point the check at the sub-package, or (b) enhance the script to skip `src/*/__init__.py` files lacking `__all__` and search one level deeper. Do not change the doc to satisfy the script.
- **Blast radius:** doc scope decision only; optional script enhancement in the skill repo.

## Name collision: `PreprocessingDefinition`

- **Current:** exported by BOTH `dr_code.eval.lifecycle` (variable-bearing kernel definition, hashed via the full Identity Hash) and `dr_code.preprocessing` (step-based runner definition, hashed via `preprocessing_definition_hash` → `trace.stable_hash`, BLAKE2b).
- **Problem:** two different concepts, one name. A "Preprocessing Definition" term row cannot map to a single exported name.
- **Proposal:** rename the runner one, e.g. `PreprocessingFlowDefinition` or `StepPipelineDefinition`. Trade-off: the runner name is older/more used; the kernel name is the contract term, so renaming the runner keeps the contract vocabulary stable.
- **Blast radius:** `preprocessing` package (`definition.py`, registry, runner, `bind_*`/`run_preprocessing`), fixtures, tests. Verify no serialized schema/hash change.

## Name collision: `CodeArtifact`

- **Current:** exported by BOTH `dr_code.eval.code` (compile-validating kernel artifact) and `dr_code.trace.artifacts` (boundary-contract artifact). Both surface at their subpackage roots.
- **Problem:** same name, two concepts.
- **Proposal:** disambiguate the eval one, e.g. `CompiledCodeArtifact`. Trade-off: the kernel term "Code Artifact" is defined by its compile-validity guarantee, so a `Compiled…` prefix reads naturally; alternatively rename the trace one.
- **Blast radius:** `eval.code`, `trace.artifacts`, corpus projection, tests, goldens. Verify no serialized schema/hash change.

## Name collision: `MetricRecord`

- **Current:** exported by BOTH `dr_code.eval.facts` and `dr_code.metrics.records`.
- **Problem:** same name, parallel representations of the same concept in two packages.
- **Proposal:** consolidate onto the kernel `MetricRecord` and retire the metrics-package one, or rename one pending consolidation. The contract crosswalk marks the `metrics` package's `MetricsDefinition`/`MetricRecord` path as a current-partial to migrate into the eval kernel.
- **Blast radius:** `metrics` package, callers, fixtures, tests.

## Parallel representations: metrics vs eval-kernel definitions

- **Current:** `dr_code.metrics` exports `MetricQuestion`, `MetricsDefinition`, `metrics_definition_hash`; `dr_code.eval.lifecycle` exports `MetricQuestionBinding`, `MetricExtractionDefinition`, `MetricExtractionConfig`.
- **Problem:** two parallel representations of metric extraction; the kernel path is the contract, the metrics-package path is a current-partial to migrate.
- **Proposal:** rename/consolidation, not a doc-side alias. Migrate `MetricsDefinition`/`MetricQuestion` onto the kernel `MetricExtractionDefinition`/`MetricQuestionBinding`.
- **Blast radius:** `metrics` package, callers, fixtures, tests.

## Near-sibling names: `CodeCandidateSet` vs `CodeCandidateSetArtifact`

- **Current:** `eval` exports `CodeCandidateSet`; `trace` exports `CodeCandidateSetArtifact` (also `IdentifiedCandidateSetArtifact`).
- **Problem:** different names for related concepts across the eval/trace boundary; the relationship is not obvious from the names.
- **Proposal:** align the naming so the trace artifact reads as the serialized/boundary form of the eval set (e.g. keep the `…Artifact` suffix convention consistently and document the mapping), or unify.
- **Blast radius:** `trace.artifacts`, corpus, tests.

## Asymmetric hashing siblings: `identity_hash_for` vs `…_definition_hash`/`stable_hash`

- **Current:** the eval kernel uses `identity_hash_for` and `Definition.identity_hash()` — the full 64-character SHA-256 Identity Hash via the storage layer (`eval/identity.py`). But `preprocessing_definition_hash` and `metrics_definition_hash` both delegate to `dr_code.trace.stable_hash`, a DIFFERENT family: `hashlib.blake2b(...).hexdigest()` over sorted-key `json.dumps`.
- **Problem:** two hash functions named similarly (`…definition_hash` vs `identity_hash`) for the same conceptual role — definition identity — but with different algorithms and output shapes. The contract requires the full Identity Hash for these definitions.
- **Proposal:** unify on the Identity Hash protocol; rename or retire `stable_hash` and the `…_definition_hash` wrappers, or clearly mark them as non-identity content hashes.
- **Blast radius:** definition identities, run-id labels, fixtures, goldens. **Fixture-regenerating — do NOT change values in a doc pass.**

## `stable_hash` (trace) — jargon-adjacent, non-identity content hash

- **Current:** `dr_code.trace.stable_hash` — a generic "content hash for frozen definitions" (BLAKE2b); its docstring cross-references `synthetic.dataset_builder._seed_for`.
- **Problem:** the name collides conceptually with the contract's Content Hash / Identity Hash distinctions; a reader cannot tell from the name that it is a non-identity hash.
- **Proposal:** rename to signal a non-identity content hash (e.g. `content_hash` / `sweep_hash`) or remove from the public surface if only internal.
- **Blast radius:** `trace`, `preprocessing`, `metrics`, `synthetic`, tests, fixtures.

## Legacy HumanEval submission scoring vs kernel Score/MetricFact

- **Current:** `dr_code.humaneval` exports `score_humaneval_submission`, `HumanEvalSubmissionScore`, `evaluation_aggregate_metrics`, `CompletedScore`, `CompletedCandidateScore`, `HumanEvalCandidateScore`, alongside the kernel `Score`/`MetricFact` path.
- **Problem:** the crosswalk marks legacy HumanEval submission scoring as supporting any-passing correctness but not authoritative. Two scoring representations coexist.
- **Proposal (contract-gap list):** either clearly mark these as pre-kernel/legacy or migrate onto `MetricFact`/`Score`.
- **Blast radius:** `humaneval` package, callers, tests, goldens.

## Minor / note-only

- **`DefinitionRef`** (`eval.lifecycle`): the contract term is "Definition Reference"; the `Ref` abbreviation is mild jargon relative to the spelled-out term. Consider `DefinitionReference`. Low priority; blast radius is every config type plus fixtures/goldens (identity-bearing field name), so treat as fixture-regenerating.
- **`resolved_operator_version` / `resolved_step_version`** (`eval.lifecycle`): function names read as values. Note only; rename to `resolve_*` if a term row ever needs them as verbs.
- **Compression reference exports** use both `…Key` and `…Artifact` plus a `…Resolver` — consistent and fine as-is.
- **`ZERO_DENOMINATOR`** is exported as a `None`-typed sentinel. The sentinel-as-export is fine; documented as the explicit zero-denominator state, not a missing value. No change.
