import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StatusBadge } from "../src/status-badge.js";

describe("StatusBadge", () => {
  it("renders its label with the status class", () => {
    render(<StatusBadge status="positive">passed</StatusBadge>);
    const badge = screen.getByText("passed");
    expect(badge.className).toContain("drv-status-badge");
    expect(badge.className).toContain("drv-status-badge-positive");
  });

  it("defaults to the neutral status and appends a custom className", () => {
    render(<StatusBadge className="extra">idle</StatusBadge>);
    const badge = screen.getByText("idle");
    expect(badge.className).toContain("drv-status-badge-neutral");
    expect(badge.className).toContain("extra");
  });
});
