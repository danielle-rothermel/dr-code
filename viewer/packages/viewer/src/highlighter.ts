import {
  createHighlighterCore,
  createJavaScriptRegexEngine,
} from "react-shiki/core";

export type ClientHighlighter = Awaited<
  ReturnType<typeof createHighlighterCore>
>;

// Explicit, boring list of grammars every client code panel can highlight.
// This is the source of truth for CodeBlock's `lang` prop: anything not
// listed here renders as plaintext (see resolveSupportedLang below).
// Kept as a fixed list rather than the full shiki bundle or dynamic
// per-render loading so the bundle stays small and predictable.
export const SUPPORTED_LANGUAGES = [
  "python",
  "typescript",
  "javascript",
  "tsx",
  "jsx",
  "json",
  "yaml",
  "toml",
  "bash",
  "markdown",
  "html",
  "css",
  "diff",
] as const;

export type SupportedLanguage = (typeof SUPPORTED_LANGUAGES)[number];

// Module-level singleton so every client code panel shares one engine
// and one grammar/theme load. Import specifiers must stay literal for
// bundler static analysis; they mirror SHIKI_THEMES in themes.ts.
let highlighterPromise: Promise<ClientHighlighter> | null = null;

export function getClientHighlighter(): Promise<ClientHighlighter> {
  highlighterPromise ??= createHighlighterCore({
    themes: [
      import("@shikijs/themes/github-light"),
      import("@shikijs/themes/github-dark"),
    ],
    langs: [
      import("@shikijs/langs/python"),
      import("@shikijs/langs/typescript"),
      import("@shikijs/langs/javascript"),
      import("@shikijs/langs/tsx"),
      import("@shikijs/langs/jsx"),
      import("@shikijs/langs/json"),
      import("@shikijs/langs/yaml"),
      import("@shikijs/langs/toml"),
      import("@shikijs/langs/bash"),
      import("@shikijs/langs/markdown"),
      import("@shikijs/langs/html"),
      import("@shikijs/langs/css"),
      import("@shikijs/langs/diff"),
    ],
    engine: createJavaScriptRegexEngine(),
  });
  return highlighterPromise;
}

/**
 * Resolves a caller-supplied `lang` against the languages actually
 * registered in `highlighter`, falling back to `"plaintext"` for
 * anything unregistered (e.g. a typo or a grammar outside
 * SUPPORTED_LANGUAGES). Keeps CodeBlock's fallback graceful and silent
 * instead of relying on react-shiki's own (undocumented) handling of
 * prebuilt highlighters.
 */
export function resolveSupportedLang(
  lang: string,
  highlighter: ClientHighlighter,
): string {
  return highlighter.getLoadedLanguages().includes(lang) ? lang : "plaintext";
}
