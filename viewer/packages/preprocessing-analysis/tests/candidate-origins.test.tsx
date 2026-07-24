import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { CandidateOrigins } from "../src/candidate-origins";

afterEach(cleanup);

describe("CandidateOrigins", () => {
  it("renders converged origins as distinct ordered operation paths", () => {
    const { container } = render(<CandidateOrigins origins={[
      {
        path: [
          { details: { name: "normalized_raw_response" }, kind: "response_representation" },
          { details: { index: 0, tag: "json" }, kind: "fenced_block" },
          { details: {}, kind: "fenced_json_code" },
        ],
      },
      {
        path: [
          { details: {}, kind: "top_level_json_code" },
          { details: {}, kind: "markdown_wrapper" },
        ],
      },
    ]} />);

    expect(screen.getByText("2 converged extraction paths")).toBeTruthy();
    const paths = Array.from(container.querySelectorAll(".candidate-origin-path"));
    expect(paths).toHaveLength(2);
    expect(Array.from(paths[0]?.querySelectorAll("strong") ?? []).map(({ textContent }) => textContent)).toEqual([
      "response representation",
      "fenced block",
      "fenced json code",
    ]);
    expect(Array.from(paths[1]?.querySelectorAll("strong") ?? []).map(({ textContent }) => textContent)).toEqual([
      "top level json code",
      "markdown wrapper",
    ]);
    expect(screen.getByText("normalized_raw_response")).toBeTruthy();
    expect(screen.getByText("json")).toBeTruthy();
  });

  it("makes an empty normalized path visible instead of dropping provenance", () => {
    render(<CandidateOrigins origins={[{ path: [] }]} />);

    expect(screen.getByText("1 extraction path")).toBeTruthy();
    expect(screen.getByText("No extraction operations recorded")).toBeTruthy();
  });

  it("renders nothing when a candidate has no origins", () => {
    const { container } = render(<CandidateOrigins origins={[]} />);

    expect(container.innerHTML).toBe("");
  });
});
