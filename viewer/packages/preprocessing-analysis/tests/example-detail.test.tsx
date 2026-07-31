import { describe, expect, it } from "vitest";

import { outcomeStatus } from "../src/example-detail";

describe("outcomeStatus", () => {
  it("classifies only the current terminal outcomes", () => {
    expect(outcomeStatus("function_candidates_extracted")).toBe("success");
    expect(outcomeStatus("decoder_output_missing")).toBe("warning");
    expect(outcomeStatus("decoder_output_blank")).toBe("warning");
    expect(outcomeStatus("no_top_level_function_candidate")).toBe("failure");
    expect(outcomeStatus("fictional_function_candidate")).toBe("failure");
  });
});
