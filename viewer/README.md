# viewer/ — dr-code React component workspace

The canonical React components for code visualization across the
dr-code ecosystem (ADR 0006). One pnpm workspace, one package:
`@dr-code/viewer` at `packages/viewer`. It never ships in the Python
wheel — the wheel stays toolchain-pure.

## Stack (pinned)

- **shiki 3.x** — `@git-diff-view/shiki` hard-depends on shiki `^3`;
  one engine and one grammar/theme system across panels and diffs.
- **@git-diff-view/react + /file + /shiki** — 0.x solo-maintainer,
  pinned exact and fully wrapped so they are swappable in one place.
- **react-shiki/core** — client-tier highlighting, same shiki engine.
- React 19 peer deps; consumers are Next.js App Router.

## Two-tier architecture

| Tier | Components | Notes |
| --- | --- | --- |
| Server (RSC, zero client JS) | `<CodeBlock>`, `<TaskCard>` | shiki `codeToHtml` in async server components |
| Client | `<CodeBlockClient>`, `<TransformDiff>`, `<ExtractionTraceView>` | `<TransformDiff>` takes two plain string props and computes the diff in-browser; DiffFile instances never cross the RSC boundary |
| Shared | `<EvaluationCaseTable>` | no client JS of its own |

Consumers import only these components, never shiki or @git-diff-view
directly.

## Consuming

Install as a pnpm git dependency pinned to a rev:

```jsonc
// package.json
"dependencies": {
  "@dr-code/viewer": "github:danielle-rothermel/dr-code#<rev>&path:/viewer/packages/viewer"
}
```

The package builds from source on install (`prepare` runs tsc). Import
the base styles once, e.g. in the root layout:

```ts
import "@dr-code/viewer/styles.css";
```

Code panels use Fira Code with ligatures; ligatures are disabled inside
intraline-change spans of diffs, where they can mask single-character
changes. Load the Fira Code font in the consuming app.

## Types

Generated, committed under `packages/viewer/src/gen/`:

- `gen/serve.ts` — from the serve OpenAPI schema
  (`pnpm gen:serve`, runs `uv run python -m dr_code.serve openapi`).
- `gen/humaneval.ts` — from the library pydantic JSON-schema dump
  (`pnpm gen:humaneval`, runs `uv run python -m dr_code.schemas humaneval`).

Regenerate with `pnpm gen` from `packages/viewer` (needs `uv` and the
repo's Python env; maintainer-only).

## Checks

```bash
pnpm install --frozen-lockfile
pnpm typecheck
pnpm build
pnpm test
```
