import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { CodeBlock } from "../src/code-block.js";

const PYTHON_SNIPPET = "def add(a, b):\n    return a + b\n";

describe("CodeBlock", () => {
  it("renders shiki-highlighted python on the server tier", async () => {
    render(await CodeBlock({ code: PYTHON_SNIPPET }));
    const pre = document.querySelector(".drv-code-block pre.shiki");
    expect(pre).not.toBeNull();
    expect(pre?.textContent).toContain("def add");
  });
});
