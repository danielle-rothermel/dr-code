import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { EvaluationCaseTable } from "../src/evaluation-case-table.js";
import type { EvaluationCaseSummary } from "../src/types.js";
import casesFixture from "./fixtures/cases.json" with { type: "json" };

const cases = casesFixture as unknown as EvaluationCaseSummary[];

describe("EvaluationCaseTable", () => {
  it("renders one row per case with status and reprs", () => {
    render(<EvaluationCaseTable cases={cases} />);
    expect(screen.getByText("case-0")).toBeDefined();
    expect(screen.getByText("case-1")).toBeDefined();
    expect(screen.getByText("passed")).toBeDefined();
    expect(screen.getByText("failed")).toBeDefined();
    expect(screen.getByText("expected 5, got 4")).toBeDefined();
  });
});
