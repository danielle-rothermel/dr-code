import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { TaskCard } from "../src/task-card.js";
import type { HumanEvalTask } from "../src/types.js";
import taskFixture from "./fixtures/task.json" with { type: "json" };

const task = taskFixture as unknown as HumanEvalTask;

describe("TaskCard", () => {
  it("renders task metadata and zero-JS code panels", async () => {
    render(await TaskCard({ task }));
    expect(screen.getByText("HumanEval/0")).toBeDefined();
    expect(
      document.querySelector(".drv-task-header code")?.textContent,
    ).toBe("add");
    expect(screen.getByText("fixture task")).toBeDefined();
    expect(
      document.querySelectorAll(".drv-task-card pre.shiki").length,
    ).toBe(3);
  });
});
