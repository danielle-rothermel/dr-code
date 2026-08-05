# `viewer/` — React code-visualization primitives

This pnpm workspace contains the publishable `@dr-code/viewer` package and its
private visual gallery. The package provides domain-agnostic React primitives;
it is not included in the Python wheel.

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
when loading completes. Content, filenames, language, and theme changes to
`CodeDiff` immediately restore the fallback until the corresponding highlighted
output is ready; split/unified mode changes render immediately from the prepared
diff. `CodeBlock` likewise restores its fallback after source, language, or
theme changes, so stale source is never shown.
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

The recursive typecheck and build commands cover both the publishable package
and the gallery. After component or style changes, also run the gallery and
inspect both theme columns in a browser.
