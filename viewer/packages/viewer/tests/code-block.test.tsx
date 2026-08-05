import { render, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { CodeBlock } from "../src/code-block.js";

const PYTHON_SNIPPET = "def double(x):\n    return x * 2\n";
const TYPESCRIPT_SNIPPET = "const answer: number = 42;\n";

describe("CodeBlock", () => {
  it("shows plain code immediately, then Shiki markup", async () => {
    const { container } = render(<CodeBlock code={PYTHON_SNIPPET} />);
    expect(within(container).getByText(/def double/)).toBeDefined();
    expect(
      container.querySelector(".drv-code-block pre:not(.shiki)"),
    ).not.toBeNull();
    await waitFor(() => {
      expect(container.querySelector(".drv-code-block pre.shiki")).not.toBeNull();
    });
    expect(container.querySelector(".drv-code-block")?.textContent).toContain(
      "def double",
    );
  });

  it("highlights a bundled non-Python language", async () => {
    const { container } = render(
      <CodeBlock code={TYPESCRIPT_SNIPPET} lang="typescript" />,
    );

    await waitFor(() => {
      expect(
        container.querySelector(".drv-code-block pre.shiki"),
      ).not.toBeNull();
    });
    expect(
      container.querySelectorAll(".drv-code-block pre.shiki span[style]")
        .length,
    ).toBeGreaterThan(0);
    expect(container.querySelector(".drv-code-block")?.textContent).toContain(
      "const answer: number = 42;",
    );
  });

  it("shows the current fallback while updated code is highlighted", async () => {
    const { container, rerender } = render(
      <CodeBlock code={PYTHON_SNIPPET} />,
    );
    await waitFor(() => {
      expect(container.querySelector(".drv-code-block pre.shiki")).not.toBeNull();
    });

    rerender(<CodeBlock code={TYPESCRIPT_SNIPPET} lang="typescript" />);

    const fallback = container.querySelector(
      ".drv-code-block pre:not(.shiki) code",
    );
    expect(fallback?.textContent).toBe(TYPESCRIPT_SNIPPET);
    expect(container.textContent).not.toContain("def double");
    await waitFor(() => {
      expect(container.querySelector(".drv-code-block pre.shiki")).not.toBeNull();
      expect(container.textContent).toContain("const answer: number = 42;");
    });
  });
});
