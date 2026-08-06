import { describe, expect, it } from "vitest";

import { getHighlighter } from "../src/highlighter.js";

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

      expect(() =>
        highlighter.codeToHtml(code, { lang, theme: "github-light" }),
      ).not.toThrow();
    },
  );
});
