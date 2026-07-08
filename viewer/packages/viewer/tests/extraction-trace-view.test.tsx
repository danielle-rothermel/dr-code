import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ExtractionTraceView } from "../src/extraction-trace-view.js";
import type { ExtractionTrace } from "../src/types.js";
import traceFixture from "./fixtures/extraction-trace.json" with { type: "json" };

const trace = traceFixture as unknown as ExtractionTrace;

describe("ExtractionTraceView", () => {
  it("renders profile, rationale, tree nodes, and the selection walk", async () => {
    render(<ExtractionTraceView trace={trace} />);
    expect(screen.getByText("humaneval-best-effort@v1")).toBeDefined();
    expect(screen.getByText(/candidate 0 selected/)).toBeDefined();
    expect(screen.getByText("normalize_text")).toBeDefined();
    expect(screen.getAllByText("compile_validation").length).toBeGreaterThan(
      0,
    );
    expect(screen.getByText("selected")).toBeDefined();
    // Wait for the client tiers to finish their async highlighting so
    // no React work is scheduled after environment teardown.
    await waitFor(
      () => {
        expect(
          document.querySelector(".drv-transform-diff-pending"),
        ).toBeNull();
        expect(
          document.querySelectorAll("pre.shiki").length,
        ).toBeGreaterThan(0);
      },
      { timeout: 10_000 },
    );
  });
});
