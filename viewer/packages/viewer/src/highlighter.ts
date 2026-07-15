import {
  createHighlighterCore,
} from "shiki/core";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";

export type Highlighter = Awaited<
  ReturnType<typeof createHighlighterCore>
>;

// Module-level singleton so every code panel shares one engine and one
// grammar/theme load. Import specifiers must stay literal for bundler
// static analysis; theme imports mirror SHIKI_THEMES in themes.ts.
let highlighterPromise: Promise<Highlighter> | null = null;

export function getHighlighter(): Promise<Highlighter> {
  highlighterPromise ??= createHighlighterCore({
    themes: [
      import("@shikijs/themes/github-light"),
      import("@shikijs/themes/github-dark"),
    ],
    langs: [
      import("@shikijs/langs/python"),
      import("@shikijs/langs/javascript"),
      import("@shikijs/langs/typescript"),
      import("@shikijs/langs/json"),
      import("@shikijs/langs/bash"),
    ],
    engine: createJavaScriptRegexEngine(),
  });
  return highlighterPromise;
}
