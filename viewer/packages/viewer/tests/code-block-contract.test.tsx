import { act, render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Highlighter } from "../src/highlighter.js";

const highlighterMocks = vi.hoisted(() => ({
  getHighlighter: vi.fn(),
}));

vi.mock("../src/highlighter.js", () => ({
  getHighlighter: highlighterMocks.getHighlighter,
}));

import { CodeBlock } from "../src/code-block.js";

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((onResolve) => {
    resolve = onResolve;
  });
  return { promise, resolve };
}

function controlledHighlighter() {
  const codeToHtml = vi.fn(
    (code: string) => `<pre class="controlled"><code>${code}</code></pre>`,
  );
  return {
    codeToHtml,
    value: { codeToHtml } as unknown as Highlighter,
  };
}

describe("CodeBlock contract", () => {
  beforeEach(() => {
    highlighterMocks.getHighlighter.mockReset();
  });

  it("preserves its base class when adding a caller class", () => {
    const pending = deferred<Highlighter>();
    highlighterMocks.getHighlighter.mockReturnValue(pending.promise);

    const { container } = render(
      <CodeBlock code="print('ready')" className="result-code" />,
    );

    expect(container.firstElementChild?.className).toBe(
      "drv-code-block result-code",
    );
  });

  it("only highlights the current props after a deferred load", async () => {
    const pending = deferred<Highlighter>();
    const highlighter = controlledHighlighter();
    highlighterMocks.getHighlighter.mockReturnValue(pending.promise);

    const { container, rerender } = render(
      <CodeBlock code="const answer = 1;" lang="javascript" />,
    );
    rerender(
      <CodeBlock
        code="const answer: number = 2;"
        lang="typescript"
        theme="dark"
      />,
    );

    expect(container.querySelector("code")?.textContent).toBe(
      "const answer: number = 2;",
    );

    await act(async () => {
      pending.resolve(highlighter.value);
      await pending.promise;
    });

    expect(highlighter.codeToHtml).toHaveBeenCalledOnce();
    expect(highlighter.codeToHtml).toHaveBeenCalledWith(
      "const answer: number = 2;",
      { lang: "typescript", theme: "github-dark" },
    );
    expect(container.querySelector("code")?.textContent).toBe(
      "const answer: number = 2;",
    );
  });

  it("does not highlight after unmounting during a deferred load", async () => {
    const pending = deferred<Highlighter>();
    const highlighter = controlledHighlighter();
    highlighterMocks.getHighlighter.mockReturnValue(pending.promise);

    const { unmount } = render(<CodeBlock code="print('ready')" />);
    unmount();

    await act(async () => {
      pending.resolve(highlighter.value);
      await pending.promise;
    });

    expect(highlighter.codeToHtml).not.toHaveBeenCalled();
  });
});
