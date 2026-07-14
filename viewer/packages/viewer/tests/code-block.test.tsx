import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { CodeBlock } from "../src/code-block.js";

const PYTHON_SNIPPET = "def double(x):\n    return x * 2\n";

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
});
