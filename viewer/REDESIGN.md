# Viewer redesign plan

Status: planned, not started. This doc is the working plan for
simplifying `@dr-code/viewer`; delete it when the work lands.

## Motivation

The viewer package predates the current direction of the repo (its ADRs
have been removed intentionally). Two properties make it harder to keep
simple and predictable than it should be:

1. **Two rendering tiers.** Server (RSC, async) and client variants of
   the same idea (`CodeBlock` vs `CodeBlockClient`) drag in the most
   confusing part of modern React — the RSC boundary — for benefits
   (zero-JS panels, no highlight flash) that don't matter for the
   package's actual job: assembling simple demos quickly.
2. **Domain coupling.** The package exports whetstone/dr-code domain
   types (`ExtractionTrace`, `HumanEvalTask`, `EvaluationCaseSummary`),
   commits generated types from the Python package's schemas
   (`src/gen/`), and hardcodes domain logic in `ExtractionTraceView`
   and `TaskCard`. Every parallel refactor of the Python package forces
   a sync here.

## Target design

A small library of **domain-agnostic, client-safe primitives**. Props
are aggressively boring: strings, numbers, small enums, `children`.
Composition into domain views (trace viewers, task cards, demo pages)
is the caller's job, not the library's.

- **One tier.** No async server components, no `CodeBlock` /
  `CodeBlockClient` split. Components are plain synchronous functions
  by default; hooks only where something genuinely loads or changes
  (the shiki highlighter, the diff). Everything renders in plain
  react-dom, Vite, and inside RSC apps as client components. Consumers
  who want server-side highlighting implement it themselves.
- **No domain types.** The public API mentions no whetstone/dr-code
  concepts. The `src/gen/` pipeline (serve OpenAPI, HumanEval schema)
  moves out of this package entirely; schema knowledge lives with the
  callers.
- **Proposed primitive set** (final names/props decided during
  implementation):
  - `CodeBlock` — highlighted code panel (today's `CodeBlockClient`,
    renamed; plain-`<pre>` fallback while the highlighter loads).
  - `CodeDiff` — before/after diff of two plain strings (today's
    `TransformDiff`, renamed to drop the domain word "transform").
  - `StatusBadge` — small pass/fail/skip-style badge with a generic
    status enum (extracted from the case table / trace view).
  - `DataTable` or similar — generic column-driven table replacing the
    hardcoded `EvaluationCaseTable`, if a generic version pulls its
    weight; otherwise drop the table and let callers write `<table>`.
  - `Panel` / `LabeledSection` — the labeled-card chrome currently
    baked into `TaskCard` and the trace view, if it earns its keep.
- **Deleted, not ported:** `ExtractionTraceView`, `TaskCard`,
  `EvaluationCaseTable` (as a domain component), all exported domain
  types, `src/gen/` and the `gen:*` scripts. Their rendering ideas
  survive as primitives; their composition moves to future demo pages.

## Steps

Each step lands as its own commit (or small series) with checks green.

1. **Collapse to one tier.** Delete server-tier `CodeBlock` and
   `TaskCard`; rename `CodeBlockClient` → `CodeBlock`; rename
   `TransformDiff` → `CodeDiff`. Update exports, styles, tests, README
   (remove the two-tier table and stale ADR references).
2. **Decouple from the domain.** Delete `ExtractionTraceView` and its
   types; extract `StatusBadge` (and any other atom worth keeping)
   from the deleted components; replace or drop `EvaluationCaseTable`;
   remove `src/gen/`, `schemas/`, and the `gen:*` scripts; purge domain
   types from `types.ts` and `index.ts`. After this step the package
   has zero knowledge of the Python package.
3. **Component viewer page.** Add a private workspace app (e.g.
   `packages/gallery`, Vite + React, `"private": true`, never
   published) that renders every exported primitive in a grid with
   hand-written fixture data: several languages, long/short code, an
   empty diff, every status value, light and dark. This is the visual
   verification surface until a real demo exists.
4. **Docs pass.** Rewrite `viewer/README.md` around the new API:
   primitive list, boring-props rule, "composition belongs to callers,"
   and the gallery as the way to eyeball changes.

Deferred until dr-code / whetstone-ai shapes settle: a real demo page
that composes the primitives against actual traces and tasks. Until
then, the gallery's fixture data stands in for it, and any "this would
be convenient for the trace view" prop additions wait for a real
caller.

## Verification

- **Every step:** `pnpm install --frozen-lockfile && pnpm typecheck &&
  pnpm build && pnpm test` from `viewer/` stays green
  (`scripts/pre-check.sh` gate).
- **Unit tests** (vitest + testing-library + jsdom) updated alongside
  each component: renamed components keep their behavioral tests
  (fallback-then-highlight for `CodeBlock`, string-props-only for
  `CodeDiff`); new atoms get their own; deleted components' tests are
  deleted with them.
- **Visual check:** run the gallery dev server and eyeball the grid in
  light and dark after any component change. Optionally add a
  Playwright screenshot script later if manual checking gets old.
- **Consumer smoke test:** the package is consumed as a pnpm git dep
  that builds on install via `prepare`; after step 2, verify a scratch
  app can install the rev, import each primitive, and render with no
  RSC framework present.
- **API surface check:** after step 2, `index.ts` (and the built
  `.d.ts`) must contain no domain-named exports — grep for
  `Extraction|HumanEval|Evaluation|Candidate|Trace` as a cheap gate.
