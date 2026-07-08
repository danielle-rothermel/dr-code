# Code rendering standard: shiki v3 + @git-diff-view, wrapped, two-tier

All code visualization across this repo's ecosystem renders through
dr-code's React component package, which standardizes on **shiki v3.x**
for syntax-highlighted display and **@git-diff-view/react +
@git-diff-view/shiki** for diffs. shiki is pinned to v3 because the diff
plugin hard-depends on it — one engine and one grammar-and-theme system
across panels and diffs matters more than the newest shiki major. Code
panels use Fira Code with ligatures (disabled inside intraline-change
spans, where ligatures can mask single-character diffs). Diffs are
computed from two raw strings via `generateDiffFile(oldName, oldContent,
newName, newContent, oldLang, newLang)` — the shape the extraction
trace's before/after cleaning steps need — never from unified-diff
patches.

The package exposes an explicit two-tier architecture. Static panels are
React Server Components (shiki `codeToHtml` server-side) and ship zero
client JavaScript — that claim is scoped to static panels. Diffs and
live-fetched text are client components (`react-shiki/core`, same
engine); `<TransformDiff>` computes its diff in-browser from two string
props, because DiffFile instances cannot cross the RSC boundary.

Consumers import dr-code's components (`<CodeBlock>`, `<TransformDiff>`,
…), never the underlying primitives, so the primitives are swappable in
one place. This wrapping is load-bearing: @git-diff-view is a 0.x
solo-maintainer package (pinned), and the mature alternative (CodeMirror
read-only + @codemirror/merge) was rejected because it is client-only
for *all* rendering with a second theme system, forfeiting zero-JS
static panels entirely. Research with primary sources:
`docs/research/react-code-visualization.md` (2026-07-08; two facts it
missed — the plugin's shiki-v3 dependency and the real generateDiffFile
arity — are corrected here after npm verification).
