import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@dr-code/viewer", () => ({
  CodeBlock: ({ code }: { code: string }) => <pre>{code}</pre>,
  CodeDiff: () => <div>diff</div>,
  StatusBadge: ({ children }: { children: ReactNode }) => <span>{children}</span>,
}));

import type { ExampleQuery } from "../src/api";
import { ExamplesPanel } from "../src/examples-panel";
import { detail, fakeApi } from "./fixtures";

afterEach(cleanup);

describe("ExamplesPanel", () => {
  it("requests offset zero immediately when the run changes", async () => {
    const getExamples = vi.fn(async (_runId: string, query: ExampleQuery) => ({
      items: [{
        annotation_verdict: null,
        context: {},
        outcome: detail.outcome,
        raw_preview: detail.raw_decoder_output,
        sample_id: detail.sample_id,
      }],
      limit: query.limit ?? 25,
      offset: query.offset ?? 0,
      total: 30,
    }));
    const api = fakeApi({ getExamples });
    const query = { stage_id: "output_nonblank" };
    const view = render(
      <ExamplesPanel
        api={api}
        selection={{ query, runId: "baseline", title: "Examples" }}
      />,
    );
    await screen.findByRole("heading", { name: detail.sample_id });
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() => expect(getExamples).toHaveBeenCalledWith(
      "baseline",
      { ...query, limit: 25, offset: 25 },
    ));

    view.rerender(
      <ExamplesPanel
        api={api}
        selection={{ query, runId: "candidate", title: "Examples" }}
      />,
    );
    await waitFor(() => expect(getExamples).toHaveBeenCalledWith(
      "candidate",
      { ...query, limit: 25, offset: 0 },
    ));
    expect(getExamples).not.toHaveBeenCalledWith(
      "candidate",
      { ...query, limit: 25, offset: 25 },
    );
  });
});
