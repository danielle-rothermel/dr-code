import { render, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { CodeDiff } from "../src/code-diff.js";

const OLD_CONTENT = "def f():\n    return 1\n";
const NEW_CONTENT = "def f():\n    return 2\n";

describe("CodeDiff", () => {
  it(
    "computes and renders a diff from two plain strings",
    async () => {
      render(<CodeDiff oldContent={OLD_CONTENT} newContent={NEW_CONTENT} />);
      await waitFor(
        () => {
          expect(document.querySelector(".drv-code-diff-pending")).toBeNull();
        },
        { timeout: 10_000 },
      );
      const container = document.querySelector(".drv-code-diff");
      expect(container?.textContent).toContain("return 1");
      expect(container?.textContent).toContain("return 2");
      expect(
        container
          ?.querySelector<HTMLElement>(".diff-style-root")
          ?.style.getPropertyValue("--diff-font-size--"),
      ).toBe("11px");
    },
    15_000,
  );

  it(
    "uses the smaller font size in split mode",
    async () => {
      const { container } = render(
        <CodeDiff
          oldContent={OLD_CONTENT}
          newContent={NEW_CONTENT}
          mode="split"
        />,
      );
      await waitFor(
        () => {
          expect(container.querySelector(".drv-code-diff-pending")).toBeNull();
        },
        { timeout: 10_000 },
      );
      expect(
        container
          .querySelector<HTMLElement>(".diff-style-root")
          ?.style.getPropertyValue("--diff-font-size--"),
      ).toBe("10.5px");
    },
    15_000,
  );

  it("shows current new content while an updated diff is built", async () => {
    const { container, rerender } = render(
      <CodeDiff oldContent={OLD_CONTENT} newContent={NEW_CONTENT} />,
    );
    await waitFor(
      () => {
        expect(container.querySelector(".drv-code-diff-pending")).toBeNull();
      },
      { timeout: 10_000 },
    );

    rerender(
      <CodeDiff
        oldContent="const answer = 1;"
        newContent="const answer = 2;"
        lang="javascript"
      />,
    );

    const fallback = container.querySelector(".drv-code-diff-pending code");
    expect(fallback?.textContent).toBe("const answer = 2;");
    expect(container.textContent).not.toContain("return 1");
    await waitFor(
      () => {
        expect(container.querySelector(".drv-code-diff-pending")).toBeNull();
        expect(container.textContent).toContain("const answer = 1;");
        expect(container.textContent).toContain("const answer = 2;");
      },
      { timeout: 10_000 },
    );
  }, 15_000);
});
