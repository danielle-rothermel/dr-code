import {
  createHighlighterCore,
  createJavaScriptRegexEngine,
} from "react-shiki/core";

export type ClientHighlighter = Awaited<
  ReturnType<typeof createHighlighterCore>
>;

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
    langs: [import("@shikijs/langs/python")],
    engine: createJavaScriptRegexEngine(),
  });
  return highlighterPromise;
}
