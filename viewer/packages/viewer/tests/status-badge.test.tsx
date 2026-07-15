import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StatusBadge } from "../src/status-badge.js";

describe("StatusBadge", () => {
  it.each(["success", "failure", "warning", "neutral"] as const)(
    "renders caller content with the %s status",
    (status) => {
      render(<StatusBadge status={status}>{status} label</StatusBadge>);

      const badge = screen.getByText(`${status} label`);
      expect(badge.dataset.status).toBe(status);
      expect(badge.classList.contains("drv-status-badge")).toBe(true);
      expect(badge.dataset.theme).toBe("light");
    },
  );

  it("marks dark presentation explicitly", () => {
    render(
      <StatusBadge status="success" theme="dark">
        ready
      </StatusBadge>,
    );

    expect(screen.getByText("ready").dataset.theme).toBe("dark");
  });
});
