# viewer/ — small React primitives for code visualization

A lightweight library of domain-agnostic React components for
displaying code and diffs. One pnpm workspace, one package: `@dr-code/viewer`
at `packages/viewer`. It never ships in the Python wheel — the wheel stays
toolchain-pure.

## Primitives

**`CodeBlock`** — Highlighted code panel with fallback. Renders syntax-highlighted
code in a `<pre>` tag; while the highlighter loads, renders plain text. Language is
matched against `SUPPORTED_LANGUAGES` in `src/highlighter.ts`; unknown languages
render as plaintext. Key props: `code` (string), `lang` (string, optional, matched
against the supported list).

**`CodeDiff`** — Before/after diff of two plain strings, computed in-browser.
Supports split and unified modes; light and dark themes. Key props: `oldContent`
(string), `newContent` (string), plus optional `oldName`, `newName`, `lang`,
`mode` ('split' | 'unified', default 'unified'), and `theme` ('light' | 'dark',
default 'light').

**`StatusBadge`** — Small status indicator badge. The label is passed as `children`
(ReactNode, required); `status` ('positive' | 'negative' | 'warning' | 'neutral')
is optional and defaults to 'neutral'.

## Design principles

- **Boring props.** Components take strings, numbers, small enums, and children only.
  No domain types, no complex objects.
- **No domain knowledge.** The package exports no whetstone/dr-code types; it has
  zero knowledge of schemas or domain logic. Composition of domain views (trace
  viewers, task cards, etc.) is the caller's job.
- **Client-safe by default.** All components are plain synchronous functions;
  hooks only where something genuinely loads (the highlighter) or changes (the diff).
  Works in plain react-dom/Vite and inside RSC apps as client components. Consumers
  wanting server-side highlighting implement it themselves.

## Stack (pinned)

- **shiki 3.x** — via `@git-diff-view/shiki`, one engine and one grammar/theme
  system across panels and diffs.
- **@git-diff-view/react + /file + /shiki** — 0.x solo-maintainer, pinned exact
  and fully wrapped so they are swappable in one place.
- **react-shiki/core** — client-tier highlighting, same shiki engine.
- React 19 peer deps; consumers are Next.js App Router or standalone.

Consumers import only the primitives; shiki and @git-diff-view are wrapped
implementation details.

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

## Gallery

`packages/gallery` (`@dr-code/gallery`, private, never published) is a
Vite + React app that renders every exported primitive against
hand-written fixture data — several languages, long/short code, a
changed diff, an identical-content diff, both `CodeDiff` modes, and
every `StatusBadge` status — with a light/dark toggle. It's the visual
verification surface for changes to the primitives; run it and eyeball
the grid after touching a component:

```bash
pnpm --filter @dr-code/gallery dev
```

## Checks

```bash
pnpm install --frozen-lockfile
pnpm typecheck
pnpm build
pnpm test
```

`scripts/pre-check.sh` runs the install/typecheck/build gate when pnpm
is available; CI always runs it.
