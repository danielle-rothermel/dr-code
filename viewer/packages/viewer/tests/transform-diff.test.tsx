import { render, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { TransformDiff } from "../src/transform-diff.js";

const OLD_CONTENT = "def f():\n    return 1\n";
const NEW_CONTENT = "def f():\n    return 2\n";

describe("TransformDiff", () => {
  it(
    "computes and renders a diff from two plain strings",
    async () => {
      render(
        <TransformDiff oldContent={OLD_CONTENT} newContent={NEW_CONTENT} />,
      );
      await waitFor(
        () => {
          expect(
            document.querySelector(".drv-transform-diff-pending"),
          ).toBeNull();
        },
        { timeout: 10_000 },
      );
      const container = document.querySelector(".drv-transform-diff");
      expect(container?.textContent).toContain("return 1");
      expect(container?.textContent).toContain("return 2");
    },
    15_000,
  );
});
