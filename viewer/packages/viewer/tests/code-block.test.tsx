import { render, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { CodeBlock } from "../src/code-block.js";

const PYTHON_SNIPPET = "def double(x):\n    return x * 2\n";
describe("CodeBlock", () => {
  it("renders a real highlighted code block", async () => {
    const { container } = render(<CodeBlock code={PYTHON_SNIPPET} />);
    expect(within(container).getByText(/def double/)).toBeDefined();

    await waitFor(() => {
      expect(container.querySelector(".drv-code-block pre.shiki")).not.toBeNull();
    });
    expect(container.querySelector(".drv-code-block")?.textContent).toContain(
      "def double",
    );
  });
});
