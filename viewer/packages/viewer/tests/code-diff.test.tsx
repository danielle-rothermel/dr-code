import { render, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { CodeDiff } from "../src/code-diff.js";

const OLD_CONTENT = "def f():\n    return 1\n";
const NEW_CONTENT = "def f():\n    return 2\n";

describe("CodeDiff", () => {
  it(
    "renders a real diff from two plain strings",
    async () => {
      const { container } = render(
        <CodeDiff oldContent={OLD_CONTENT} newContent={NEW_CONTENT} />,
      );
      await waitFor(
        () => {
          expect(container.querySelector(".drv-code-diff-pending")).toBeNull();
        },
        { timeout: 10_000 },
      );
      const diff = container.querySelector(".drv-code-diff");
      expect(diff?.textContent).toContain("return 1");
      expect(diff?.textContent).toContain("return 2");
    },
    15_000,
  );
});
