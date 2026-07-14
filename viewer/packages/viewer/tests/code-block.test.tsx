import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { CodeBlock } from "../src/code-block.js";

const PYTHON_SNIPPET = "def double(x):\n    return x * 2\n";
const TYPESCRIPT_SNIPPET =
  "const x: number = 1;\nfunction f(y: string) { return y; }\n";

describe("CodeBlock", () => {
  it("shows plain code immediately, then shiki markup", async () => {
    render(<CodeBlock code={PYTHON_SNIPPET} />);
    expect(screen.getByText(/def double/)).toBeDefined();
    await waitFor(() => {
      expect(
        document.querySelector(".drv-code-block pre.shiki"),
      ).not.toBeNull();
    });
    expect(document.querySelector(".drv-code-block")?.textContent).toContain(
      "def double",
    );
  });

  it("produces colored token spans for typescript once the highlighter loads", async () => {
    render(<CodeBlock code={TYPESCRIPT_SNIPPET} lang="typescript" />);
    await waitFor(() => {
      expect(
        document.querySelector(".drv-code-block pre.shiki"),
      ).not.toBeNull();
    });
    const spans = document.querySelectorAll(
      ".drv-code-block pre.shiki code span[style]",
    );
    expect(spans.length).toBeGreaterThan(1);
    const colors = new Set(
      Array.from(spans).map((span) => span.getAttribute("style")),
    );
    expect(colors.size).toBeGreaterThan(1);
  });

  it("renders code text without throwing for an unknown lang", async () => {
    render(<CodeBlock code={PYTHON_SNIPPET} lang="klingon" />);
    await waitFor(() => {
      expect(
        document.querySelector(".drv-code-block pre.shiki"),
      ).not.toBeNull();
    });
    expect(document.querySelector(".drv-code-block")?.textContent).toContain(
      "def double",
    );
  });
});
