import { describe, expect, it } from "vitest";
import { getDiffViewHighlighter } from "@git-diff-view/shiki";

import {
  getHighlighter,
  isSupportedLanguage,
  SHIKI_LANGUAGES,
} from "../src/highlighter.js";

const LANGUAGE_ALIASES = [
  ["python", "print('ready')"],
  ["py", "print('ready')"],
  ["javascript", "const answer = 42;"],
  ["js", "const answer = 42;"],
  ["cjs", "const answer = 42;"],
  ["mjs", "const answer = 42;"],
  ["typescript", "const answer: number = 42;"],
  ["ts", "const answer: number = 42;"],
  ["cts", "const answer: number = 42;"],
  ["mts", "const answer: number = 42;"],
  ["json", '{"answer": 42}'],
  ["shellscript", 'echo "$PATH"'],
  ["bash", 'echo "$PATH"'],
  ["sh", 'echo "$PATH"'],
  ["shell", 'echo "$PATH"'],
  ["zsh", 'echo "$PATH"'],
] as const;

describe("getHighlighter", () => {
  it("shares one highlighter promise", async () => {
    const first = getHighlighter();
    const second = getHighlighter();

    expect(second).toBe(first);
    await first;
  });

  it.each(LANGUAGE_ALIASES)(
    "renders the documented %s alias",
    async (lang, code) => {
      const highlighter = await getHighlighter();

      expect(isSupportedLanguage(lang)).toBe(true);
      expect(() =>
        highlighter.codeToHtml(code, { lang, theme: "github-light" }),
      ).not.toThrow();
    },
  );

  it("limits the diff highlighter to the documented grammars and themes", async () => {
    const highlighter = await getDiffViewHighlighter([...SHIKI_LANGUAGES]);
    const engine = highlighter.getHighlighterEngine();

    expect(new Set(engine?.getLoadedLanguages())).toEqual(
      new Set(LANGUAGE_ALIASES.map(([lang]) => lang)),
    );
    expect(engine?.getLoadedThemes()).toEqual(["github-light", "github-dark"]);
  });
});
