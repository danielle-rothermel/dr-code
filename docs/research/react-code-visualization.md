# React Code Visualization: Ecosystem Survey and Standard Stack Pick

**Research date:** 2026-07-08. All version numbers, publish dates, download counts, and repo stats were pulled directly from the npm registry API, the GitHub API, and official project docs on this date.

**Consumers in scope:** dr-code's future React component library (extraction traces, per-test-case eval results, cleaning-step before/after diffs, task cards with Python source), rendered inside unitbench (Next.js 16 App Router, React 19). Priorities: RSC/server rendering, read-only display, beautiful Python with Fira Code + ligatures.

---

## TL;DR — Recommended standard stack

**Primary display primitive: Shiki v4, called directly in React Server Components** (`codeToHtml` / `codeToHast` in async server components — the pattern in [Shiki's official Next.js guide](https://shiki.style/packages/next)). Wrap it in dr-code's own thin `<CodeBlock>` component. Zero client JS, VS Code-grade Python grammar, built-in [dual light/dark themes](https://shiki.style/guide/dual-themes), and Fira Code + ligatures is one CSS rule on the emitted `<pre class="shiki">`. If a client-side case ever appears (live/streaming highlight), the escape hatch is [react-shiki](https://github.com/AVGVSTVS96/react-shiki)'s `react-shiki/core` entry (~12 KB + only the langs you import) — same Shiki engine, same themes.

**Diff solution: [@git-diff-view/react](https://github.com/MrWangJustToDo/git-diff-view) + @git-diff-view/shiki.** It accepts **two raw strings** via `generateDiffFile(name, oldContent, newContent, ...)` (no unified-diff/git-patch input required), does split and unified views, wraps lines (`diffViewWrap`), ships light/dark themes, and its README explicitly claims "SSR/RSC Ready — full server-side rendering and React Server Components support." Using its Shiki highlighter plugin keeps one grammar+theme system across plain code and diffs. Caveat: still 0.x (0.1.6), single primary maintainer — pin the version.

**Runner-up combo:** CodeMirror 6 read-only ([@uiw/react-codemirror](https://github.com/uiwjs/react-codemirror)) + [@codemirror/merge](https://github.com/codemirror/merge) (`MergeView` / `unifiedMergeView` take raw doc strings). Editor-grade rendering and the most battle-tested raw-string merge view, but it is client-only JS (no RSC), themes are a separate system from Shiki, and the CodeMirror project moved off GitHub to a self-hosted Forgejo in April 2026 (solo-maintainer bus factor; npm distribution unaffected).

**Explicitly rejected for the standard:** react-syntax-highlighter (heavy, RSC support still an open issue since 2022), prism-react-renderer (no release since Dec 2024), Monaco (72 MB package, client-only, editor overkill for read-only), react-diff-viewer-continued (emotion styling → client-only), bright (dormant since Dec 2024), Code Hike (excellent but MDX-pipeline-shaped, small niche).

---

## Comparison table

| Option | Latest (date) | License / backing | RSC / zero client JS? | Raw-string diff input | Python fidelity | Fira Code + ligatures | npm dl/wk | Maintenance 2025–26 |
|---|---|---|---|---|---|---|---|---|
| **Shiki v4 (direct)** | 4.3.1 (2026-07-03) | MIT, shikijs org (13.5k★) | **Yes** — official RSC pattern | n/a (display only) | TextMate/VS Code grammar (best-in-class) | Plain HTML → one CSS rule | 14.9M | Very active; v4 major 2026-02, 53+ commits in 2026 |
| react-shiki | 0.10.1 (2026-05-25) | MIT, solo (AVGVSTVS96, 532★) | No (client hook/component) | n/a | Same as Shiki | CSS | 379k | Active (47 commits 2026) |
| prism-react-renderer | 2.4.1 (2024-12-11) | MIT, Formidable/Nearform (2k★) | No RSC story | n/a | Prism grammar (weaker for Python edge cases) | CSS | 3.7M | **Stale** — 5 commits since 2025-01 |
| react-syntax-highlighter | 16.1.1 (2026-02-26) | MIT, community org (4.7k★) | RSC support **open issue since 2022** | n/a | hljs or Prism | CSS | 6.9M | Sporadic; v16 = refractor v5 bump; 137 open issues |
| @uiw/react-codemirror (read-only) | 4.25.10 (2026-05-21) | MIT, uiwjs (2.2k★) | No — client editor | n/a | Lezer `@codemirror/lang-python` (very good) | `.cm-content` CSS | 3.5M | Active; React 19 works via peer `react>=17` |
| @monaco-editor/react (read-only) | 4.7.0 (2025-02-13); monaco 0.55.1 (2025-11-20) | MIT, Microsoft (46k★) | No — client-only, needs `ssr:false` | n/a | VS Code tokenizer | `fontLigatures: true` option | 4.8M | monaco active; react wrapper slow (last release 2025-02) |
| Code Hike | 1.1.0 (2026-03-17) | MIT, pomber (5.4k★) | Yes — async `highlight()` + `<Pre/>` | Annotation-based diff only | Shiki-family (lighter, 211 langs) | CSS | 19k | Moderate (1 release in 2026) |
| bright | 1.0.0 (2024-12-26) | MIT, pomber (1.6k★) | Yes — RSC-only | n/a | lighter | CSS | 4.4k | **Dormant** — 0 commits since 2025 |
| **@git-diff-view/react** | 0.1.6 (2026-06-21) | MIT, solo (MrWangJustToDo, 722★) | **Yes** — "SSR/RSC Ready" | **Yes** — `generateDiffFile(old, new)` | lowlight default, **Shiki plugin** | CSS (`diffViewFontSize`, plain HTML) | 101k | Very active (77 commits in 2026) |
| react-diff-viewer-continued | 4.2.2 (2026-04-23) | MIT, solo (Aeolun, 226★) | No — emotion styling, web worker | **Yes** — `oldValue`/`newValue` strings | DIY via `renderContent` render-prop | CSS | 722k | Active-ish; React 19 supported (issue #63 closed) |
| @codemirror/merge | 6.12.2 (2026-06-09) | MIT, Marijn (solo) | No — client editor | **Yes** — raw doc strings | Lezer python | `.cm-content` CSS | 1.3M | Active; repo moved to Forgejo, GitHub archived |
| Monaco diff editor | (monaco 0.55.1) | MIT, Microsoft | No | **Yes** — original/modified models | VS Code | `fontLigatures` | — | Active core |
| @shikijs/transformers (notation diff) | 4.3.1 (2026-07-03) | MIT, shikijs | Yes (RSC) | **No** — comment annotations only, not computed | Shiki | CSS | — | Active |

Sizes (npm `unpackedSize`, registry): shiki 602 KB (core subset ~106 KB, langs/themes lazy ESM); react-syntax-highlighter 2.19 MB (+ hljs + prismjs + refractor + lowlight deps); prism-react-renderer 734 KB; @uiw/react-codemirror 820 KB (+ CM6 packages); **monaco-editor 72.6 MB**; @git-diff-view/react 1.31 MB; react-diff-viewer-continued 1.10 MB; codehike 131 KB; bright 20 KB. With the recommended RSC stack, **client JS payload for plain code display is zero**.

---

## 1. Syntax-highlighted read-only display

### Shiki (recommended)

- **Current state:** v4.3.1, published 2026-07-03 ([npm registry](https://registry.npmjs.org/shiki)); MIT. v4.0.0 landed 2026-02-27 and was a cleanup-only major — Node ≥ 20, removal of deprecated typo APIs; features ship in minors ([v4 blog](https://shiki.style/blog/v4)). Releases roughly monthly through 2026 (4.1.0 May, 4.2.0 June, 4.3.0/4.3.1 June–July) ([GitHub releases](https://github.com/shikijs/shiki/releases)). 13.5k stars, 110 open issues, 50+ commits in 2026, org-backed (shikijs, core maintainer Anthony Fu). 14.9M downloads/week.
- **RSC / Next.js:** Officially documented pattern — an `async` server component awaiting `codeToHtml(...)` into `dangerouslySetInnerHTML`, or `codeToHast` + `hast-util-to-jsx-runtime` to render real elements without raw HTML ([Next.js guide](https://shiki.style/packages/next)). Caveat from the same page: avoid Edge runtime (lazy imports of langs/themes); use Node/serverless. For repeated use, create one long-lived `createHighlighter` instance.
- **Output quality:** VS Code TextMate grammars → the same Python tokenization VS Code produces, including f-strings, decorators, type hints. [Dual light/dark themes](https://shiki.style/guide/dual-themes) are first-class (`themes: { light, dark }` with CSS-variable switching). Line numbers are **not built-in** — standard approach is CSS counters on the emitted `.line` spans (or a tiny transformer); react-shiki demonstrates the CSS-based approach. Line wrapping: plain `<pre>/<code>` output, so `white-space: pre-wrap` just works.
- **Fonts:** Output is inert HTML — `pre.shiki { font-family: "Fira Code", monospace; font-variant-ligatures: contextual; }` and done. No fight with an editor's font measurement (unlike CM/Monaco).
- **Bundle:** grammars/themes/wasm are lazy-loaded ESM; `shiki/core` ~106 KB with hand-picked langs vs the full `shiki/bundle/web` ([guide](https://shiki.style/guide/)). In RSC all of it stays on the server: zero client JS.
- **Extras:** [@shikijs/transformers](https://shiki.style/packages/transformers) adds line-highlight, focus, word-highlight, whitespace rendering, indent guides — useful for annotating extraction traces.

### react-shiki (client-side escape hatch)

0.10.1 (2026-05-25), MIT, [repo](https://github.com/AVGVSTVS96/react-shiki). Exports `ShikiHighlighter` component + `useShikiHighlighter` hook — hook-based, i.e. **client** components; no RSC claim in its docs. Three entries: `react-shiki` (~1.2 MB gz, all langs), `react-shiki/web` (~707 KB gz), `react-shiki/core` (~12 KB + user-imported langs/themes). CSS-based line numbers (`showLineNumbers`, `--rs-*` variables), dual themes incl. `light-dark()` support, throttled streaming highlight via `delay`. Small but healthy: 47 commits in 2026, 3 open issues, 379k dl/week. Solo maintainer.

### prism-react-renderer — rejected

v2.4.1 published 2024-12-11 ([npm](https://registry.npmjs.org/prism-react-renderer)); 5 commits since 2025-01 in [FormidableLabs/prism-react-renderer](https://github.com/FormidableLabs/prism-react-renderer); last repo push 2025-01-02. Render-prop API with a vendored Prism; no RSC story. Effectively in maintenance freeze for ~19 months, and Prism grammars are noticeably weaker than TextMate for Python (f-string interior, decorators). Still 3.7M dl/week from older documentation stacks (Docusaurus etc.), but not the right 2026 pick.

### react-syntax-highlighter — rejected

v16.1.1 (2026-02-26); v16.0.0 (2025-10-22) was just a refractor-v5 security bump ([release notes](https://github.com/react-syntax-highlighter/react-syntax-highlighter/releases/tag/v16.0.0)). Ships hljs **and** prismjs **and** lowlight **and** refractor (2.19 MB unpacked before deps). ["Support for Server Components" #493](https://github.com/react-syntax-highlighter/react-syntax-highlighter/issues/493) has been open since Nov 2022, plus [#536 (Next 13 server components)](https://github.com/react-syntax-highlighter/react-syntax-highlighter/issues/536) — the RSC story is community workarounds. 137 open issues. No reason to prefer it over Shiki for new code.

### CodeMirror 6 read-only (@uiw/react-codemirror) — runner-up primitive

v4.25.10 (2026-05-21), MIT, [uiwjs/react-codemirror](https://github.com/uiwjs/react-codemirror). Client-only (real editor); read-only via `editable={false}` + `readOnly`. Python via `@codemirror/lang-python` 6.2.1 (Lezer parser — accurate and incremental). Peer deps allow `react >= 17`, so React 19 is permitted (their internal React 19 devDep bump [PR #706](https://github.com/uiwjs/react-codemirror/pull/706) is still open, but that's their test matrix, not the published peer range). Fonts: theme CSS on `.cm-content`. Governance note: the whole CodeMirror project (including `@codemirror/merge`, `codemirror/dev`) moved off GitHub to Marijn Haverbeke's self-hosted Forgejo at code.haverbeke.berlin, announced 2026-04-02, with GitHub repos archived; npm publishing continues normally (`@codemirror/merge` 6.12.2 shipped 2026-06-09 after the move). Right choice only if in-browser interactivity (folding, selection-driven tooling, editing) becomes a requirement.

### Monaco (@monaco-editor/react) — rejected for this use

monaco-editor 0.55.1 (2025-11-20; only `-dev` prereleases since, latest 2026-06-25), **72.6 MB unpacked**; wrapper @monaco-editor/react 4.7.0 (2025-02-13), peer react up to ^19, loads Monaco from CDN by default. Client-only — Next.js usage requires `dynamic(..., { ssr: false })` (long issue history, e.g. [#609](https://github.com/suren-atoyan/monaco-react/issues/609), [#503](https://github.com/suren-atoyan/monaco-react/issues/503), all closed with that guidance). It does have first-class `readOnly?: boolean` and `fontLigatures?: boolean | string` options (verified in [monaco.d.ts @ 0.55.1](https://unpkg.com/monaco-editor@0.55.1/monaco.d.ts), lines 3323/3429). A full IDE surface is the wrong tool for read-only display cards.

### Newer entrants

- **Code Hike v1.1.0** (2026-03-17, MIT, [code-hike/codehike](https://github.com/code-hike/codehike), 5.4k★): async `highlight()` + `<Pre/>` with an annotation-handler system; **works in RSC**; 211 langs; diff *annotations* handler ([docs](https://codehike.org/docs/concepts/code)). Genuinely maintained but designed around MDX content pipelines and authored annotations — more machinery than dr-code's programmatic rendering needs, and its diff support is annotation-based, not computed.
- **bright v1.0.0** (2024-12-26, [code-hike/bright](https://github.com/code-hike/bright)): the original "RSC-only code block" — but **zero commits since 2025**; superseded by direct Shiki-in-RSC. Skip.
- **sugar-high** 1.2.1 (2026-05-31): tiny regex highlighter aimed at JS/TS; insufficient Python fidelity. **rehype-pretty-code** 0.14.4 (2026-07-06): actively maintained Shiki wrapper, but for unified/MDX pipelines, not component APIs.

---

## 2. Diff rendering

The key requirement: **computed diffs of two arbitrary strings** (cleaning-step before/after), not git patches.

### @git-diff-view/react (recommended)

0.1.6 (2026-06-21), MIT, [MrWangJustToDo/git-diff-view](https://github.com/MrWangJustToDo/git-diff-view); 77 commits in 2026, 16 open issues, 101k dl/week. Two input modes ([README](https://github.com/MrWangJustToDo/git-diff-view)): `data={{ oldFile, newFile, hunks }}` for real git hunks, **or `generateDiffFile(fileName, oldContent, newContent, ...)`** (from `@git-diff-view/file`) which computes the diff **from two raw strings** — exactly the extraction-trace case. Split and unified views (`diffViewMode`), line wrapping (`diffViewWrap`), light/dark (`diffViewTheme`), syntax highlighting via bundled lowlight or **`@git-diff-view/shiki` 0.1.6** — which means the diff viewer can share dr-code's Shiki theme. README states "SSR/RSC Ready — full server-side rendering and React Server Components support" (SSR implementation landed in [#17](https://github.com/MrWangJustToDo/git-diff-view/issues/17), Nov 2024); a pure-CSS stylesheet (`diff-view-pure.css`) is provided. Peer deps: react ^16.8–^19. Risks: 0.x semver and effectively one maintainer — pin exact versions and wrap it behind a dr-code component so it's swappable.

### react-diff-viewer-continued — rejected

v4.2.2 (2026-04-23), MIT, [Aeolun/react-diff-viewer-continued](https://github.com/Aeolun/react-diff-viewer-continued). Does take raw strings (`oldValue`/`newValue`, `compareMethod` = jsdiff modes) with split/unified, dark theme, and virtualization — but it's styled with **emotion** (`@emotion/react` in deps), which makes it a client component by construction (emotion has no RSC support), and it runs diffing in a web worker by default (worker bundling broke ESM consumers, [#84](https://github.com/Aeolun/react-diff-viewer-continued/issues/84), fixed 2026-02). React 19 supported since [#63](https://github.com/Aeolun/react-diff-viewer-continued/issues/63) (Feb 2025). Syntax highlighting is DIY via a per-line `renderContent` render-prop. Workable, but strictly worse than @git-diff-view/react for this stack.

### @codemirror/merge — runner-up diff

6.12.2 (2026-06-09), MIT. `new MergeView({ a: { doc }, b: { doc } })` — **raw document strings**; `unifiedMergeView({ original: "..." })` for unified inline view; read-only via `EditorState.readOnly.of(true)` ([README/example](https://github.com/codemirror/merge)). Most mature raw-string merge widget in the ecosystem (1.3M dl/week), pairs naturally with the CodeMirror runner-up primitive. Client-only; GitHub repo archived after the Forgejo move (development continues at code.haverbeke.berlin; npm unaffected).

### Monaco diff editor — rejected

`renderSideBySide?: boolean` and raw original/modified models confirmed in [monaco.d.ts](https://unpkg.com/monaco-editor@0.55.1/monaco.d.ts) (line 4005). Same disqualifiers as Monaco generally: 72 MB dep, client-only, `ssr:false` dance in Next.

### Shiki notation-diff — not a diff engine

[`transformerNotationDiff`](https://shiki.style/packages/transformers) only marks lines you hand-annotate with `// [!code ++]` / `// [!code --]` comments; **it does not compute diffs between two strings**, and no @shikijs transformer does. Fine for hand-authored docs, wrong for computed before/after. (A viable DIY path — [jsdiff](https://github.com/kpdecker/jsdiff) v9.0.0, 2026-04-13, BSD-3-Clause, computes line diffs from two raw strings and you render add/remove rows through Shiki decorations — but @git-diff-view/react already productizes exactly that.)

---

## Decision rationale

1. **RSC-first wins on every axis that matters here.** These code-display surfaces are read-only. Shiki-in-RSC ships zero client JS to unitbench, and both pieces of the recommended combo render on the server in Next 16 App Router; every alternative primitive (CM6, Monaco, react-syntax-highlighter, prism-react-renderer, react-diff-viewer-continued) drags a client runtime for static output.
2. **One highlighting system end-to-end.** Shiki for code blocks + `@git-diff-view/shiki` for diffs = a single TextMate grammar and theme pair (light/dark) everywhere, and the best available Python tokenization (VS Code's own grammars).
3. **Typography is trivial where it should be.** Both recommended pieces emit plain HTML, so Fira Code + `font-variant-ligatures` is one shared CSS rule in the dr-code component library — no editor font-measurement machinery.
4. **Maintenance reality (July 2026):** shiki is org-backed with monthly releases and a clean v4; git-diff-view had 77 commits this year with SSR/RSC as an advertised feature. The 0.x/bus-factor risk on git-diff-view is contained by pinning and wrapping; the runner-up (@uiw/react-codemirror + @codemirror/merge) remains the fallback if it ever stalls, at the cost of client JS and a second theme system.
