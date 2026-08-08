import {
  createHighlighterCore,
} from "shiki/core";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";
import type { BundledLanguage } from "shiki";

export type Highlighter = Awaited<
  ReturnType<typeof createHighlighterCore>
>;

export const SHIKI_LANGUAGES = [
  "python",
  "javascript",
  "typescript",
  "json",
  "bash",
] as const satisfies readonly BundledLanguage[];

const SHIKI_LANGUAGE_ALIASES = new Set([
  "python",
  "py",
  "javascript",
  "js",
  "cjs",
  "mjs",
  "typescript",
  "ts",
  "cts",
  "mts",
  "json",
  "shellscript",
  "bash",
  "sh",
  "shell",
  "zsh",
]);

export function isSupportedLanguage(lang: string): boolean {
  return SHIKI_LANGUAGE_ALIASES.has(lang);
}

// Module-level singleton so every CodeBlock shares one engine and one
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
