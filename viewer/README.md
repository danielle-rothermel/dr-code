# `viewer/` — React code-visualization primitives

This pnpm workspace contains the publishable `@dr-code/viewer` primitives, a
private visual gallery, and the dynamic preprocessing-analysis application.
Frontend runtime data comes only from the local Python API.

## Public API

Import all components and their prop types from `@dr-code/viewer`.

### `CodeBlock`

Renders a syntax-highlighted code panel.

```tsx
<CodeBlock
  code={source}
  lang="typescript"
  theme="dark"
  className="result-code"
/>
```

- `code: string`
- `lang?: string` — defaults to `python`
- `theme?: "light" | "dark"` — defaults to `light`
- `className?: string`

### `CodeDiff`

Computes and renders a syntax-highlighted diff from two strings. Callers never
need to construct or pass a diff-library object.

```tsx
<CodeDiff
  oldContent={before}
  newContent={after}
  oldName="before.ts"
  newName="after.ts"
  lang="typescript"
  mode="split"
  theme="dark"
/>
```

- `oldContent: string`
- `newContent: string`
- `oldName?: string` — defaults to `before`
- `newName?: string` — defaults to `after`
- `lang?: string` — defaults to `python`
- `mode?: "split" | "unified"` — defaults to `unified`
- `theme?: "light" | "dark"` — defaults to `light`

### `StatusBadge`

Renders caller-provided content with a semantic status color.

```tsx
<StatusBadge status="success" theme="dark">Passed</StatusBadge>
```

- `status: "success" | "failure" | "warning" | "neutral"`
- `children: ReactNode`
- `theme?: "light" | "dark"` — defaults to `light`
- `className?: string`

The public contract stays deliberately small: strings, small unions,
`className`, and `children`. The package does not accept application-domain
objects or own their schemas. Callers compose these primitives into their own
pages and domain views.

## Languages and loading behavior

The package bundles these grammars:

| Language | Accepted `lang` values |
| --- | --- |
| Python | `python`, `py` |
| JavaScript | `javascript`, `js`, `cjs`, `mjs` |
| TypeScript | `typescript`, `ts`, `cts`, `mts` |
| JSON | `json` |
| Shell | `shellscript`, `bash`, `sh`, `shell`, `zsh` |

`CodeBlock` and `CodeDiff` are synchronous client components with no
server-only API. They can be rendered by plain React DOM or Vite applications
and used at a client boundary in an RSC application. While their highlighters
load, `CodeBlock` renders its source as plain `<pre><code>` text and `CodeDiff`
renders `newContent` the same way. The highlighted view replaces that fallback
when loading completes. Updated props immediately restore the fallback until
the corresponding highlighted output is ready, so stale source is never shown.
`StatusBadge` does not load anything asynchronously. Pass the same explicit
`theme` to each primitive when composing a light or dark surface.

## Consuming the package

Install a revision as a pnpm git dependency:

```jsonc
{
  "dependencies": {
    "@dr-code/viewer": "github:danielle-rothermel/dr-code#<rev>&path:/viewer/packages/viewer"
  }
}
```

The package builds on installation through its `prepare` script. Consumers
must provide React 19 and React DOM 19.

Import the base styles once at the application root:

```ts
import "@dr-code/viewer/styles.css";
```

The styles are required for the component layout, diff presentation, badge
colors, and bundled variable Fira Code font. Code surfaces fall back to the
system monospace stack while the font loads.

## Gallery

`@dr-code/gallery` is a private Vite app for visually checking every primitive
with static fixtures. It shows light and dark presentations, all badge states,
short and long examples in every bundled language, and changed and unchanged
diffs in both unified and split modes. Its Vite configuration resolves the
viewer package to `src/`, so component and style edits appear immediately
during development without a separate viewer build.

From `viewer/`, install dependencies and start the development server:

```bash
pnpm install --frozen-lockfile
pnpm --filter @dr-code/gallery dev
```

Build the gallery without starting a server:

```bash
pnpm --filter @dr-code/gallery build
```

## Verification

Run the complete workspace checks from `viewer/`:

```bash
pnpm install --frozen-lockfile
pnpm typecheck
pnpm build
pnpm test
```

The recursive checks cover the primitives, gallery, and preprocessing
application.

## Dynamic preprocessing application

Create one JSON descriptor per immutable run. Its fields are exact: `label`,
`dataset_id`, `corpus`, `preprocessing`, and optional
`candidate_evaluation`. `dataset_id` is the canonical dataset namespace for
every `task_id` in the corpus. When candidate evaluation is present it must
match the authenticated evaluation manifest. Relative paths are resolved from
the descriptor:

```json
{
  "label": "candidate",
  "dataset_id": "evalplus/humanevalplus",
  "corpus": "../data/corpus.parquet",
  "preprocessing": "../data/preprocessing/candidate",
  "candidate_evaluation": "../data/evaluations/candidate"
}
```

Build and serve it from the repository root:

```bash
cd viewer
pnpm install --frozen-lockfile
pnpm typecheck
pnpm test
pnpm build
cd ..

uv run dr-code viewer \
  --run viewer/runs/candidate.json \
  --database .runs/viewer.duckdb
```

The frontend workspace source and frozen `pnpm-lock.yaml` are the source of
truth for shipped assets. A normal preprocessing build writes its ignored
production output to `src/dr_code/viewer/static`. Release maintainers update
the derived immutable archive with the pinned Node 22 and pnpm 11.9.0
toolchain:

```bash
python3 scripts/build_viewer_assets.py
```

The builder installs and builds a clean copy of the workspace in a temporary
directory, stages both outputs, then replaces each artifact atomically. Verify
that the checked artifacts match current source without changing the worktree:

```bash
python3 scripts/build_viewer_assets.py --check
```

CI runs this freshness check before Python tests can validate or build wheels.
Source distributions carry that hash-checked archive. Wheel builds consume the
archive without invoking Node, pnpm, or the frontend workspace, so direct
source wheels and wheels rebuilt from an sdist contain the same package
resources. Installed wheels also need no JavaScript toolchain.

The service binds only to loopback. It validates all manifest hashes, schemas,
and row counts before registration, keeps source Parquets external, and uses
DuckDB only for run registrations, tags, and example annotations. Unknown
frontend paths receive the SPA shell; `/api` paths never do.
