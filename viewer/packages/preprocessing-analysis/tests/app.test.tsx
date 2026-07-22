import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@dr-code/viewer", () => ({
  CodeBlock: ({ code }: { code: string }) => <pre>{code}</pre>,
  CodeDiff: () => <div>code diff</div>,
  StatusBadge: ({ children }: { children: ReactNode }) => <span>{children}</span>,
}));

import { PreprocessingViewer } from "../src/app";
import { candidateRun, detail, examples, fakeApi } from "./fixtures";

afterEach(cleanup);

describe("PreprocessingViewer", () => {
  it("loads registered runs and drills into an exact waterfall stage", async () => {
    const api = fakeApi();
    render(<PreprocessingViewer api={api} />);

    expect(await screen.findByRole("heading", { name: "Trace every stage back to examples" })).toBeTruthy();
    fireEvent.click(await screen.findByRole("button", { name: "Inspect 7 examples at Nonblank output" }));

    expect(await screen.findByRole("heading", { name: "sample-1" })).toBeTruthy();
    expect(api.getExamples).toHaveBeenCalledWith("baseline", {
      limit: 25,
      offset: 0,
      stage_id: "output_nonblank",
    });
    expect(screen.getByText("No candidate survived preprocessing.")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", {
      name: "Inspect 3 examples that did not reach Nonblank output",
    }));
    await waitFor(() => expect(api.getExamples).toHaveBeenCalledWith("baseline", {
      limit: 25,
      offset: 0,
      stage_id: "lost:output_nonblank",
    }));
    expect(screen.getByRole("heading", { name: "Did not reach Nonblank output" })).toBeTruthy();
  });

  it("shows empty and backend error states", async () => {
    const { rerender } = render(<PreprocessingViewer api={fakeApi({ getRuns: vi.fn().mockResolvedValue([]) })} />);
    expect(await screen.findByRole("heading", { name: "No runs are registered" })).toBeTruthy();

    rerender(<PreprocessingViewer api={fakeApi({ getRuns: vi.fn().mockRejectedValue(new Error("service unavailable")) })} />);
    expect(await screen.findByText("service unavailable")).toBeTruthy();
  });

  it("explains an incompatible comparison and inspects compatible transitions", async () => {
    const incompatibleApi = fakeApi({ compare: vi.fn().mockRejectedValue(new Error("Corpus fingerprints differ")) });
    const { unmount } = render(<PreprocessingViewer api={incompatibleApi} />);
    await screen.findByRole("heading", { name: "Trace every stage back to examples" });
    fireEvent.click(screen.getByRole("button", { name: "Compare" }));
    expect(await screen.findByText("Corpus fingerprints differ")).toBeTruthy();
    unmount();

    const getExample = vi.fn(async (runId: string) => runId === candidateRun.run_id
      ? { ...detail, outcome: "function_candidate", raw_decoder_output: "candidate output" }
      : { ...detail, raw_decoder_output: "baseline output" });
    const getExamples = vi.fn().mockResolvedValue(examples);
    const api = fakeApi({ getExample, getExamples });
    render(<PreprocessingViewer api={api} />);
    await screen.findByRole("heading", { name: "Trace every stage back to examples" });
    fireEvent.click(screen.getByRole("button", { name: "Compare" }));

    fireEvent.click(await screen.findByRole("button", {
      name: "Inspect 8 candidate examples at Nonblank output",
    }));
    await waitFor(() => expect(getExamples).toHaveBeenCalledWith("candidate", {
      limit: 25,
      offset: 0,
      stage_id: "output_nonblank",
    }));
    await waitFor(() => expect(getExample).toHaveBeenCalledWith("candidate", "sample-1"));

    fireEvent.click(await screen.findByRole("button", { name: /compile failed.*function candidate.*1/i }));

    await waitFor(() => expect(getExamples).toHaveBeenCalledWith("baseline", {
      baseline_outcome: "compile_failed",
      candidate_outcome: "function_candidate",
      compare_run_id: candidateRun.run_id,
      limit: 25,
      offset: 0,
    }));
    await waitFor(() => {
      expect(getExample).toHaveBeenCalledWith("baseline", "sample-1");
      expect(getExample).toHaveBeenCalledWith("candidate", "sample-1");
    });
    expect(within(screen.getByRole("region", { name: "Baseline example detail" })).getByText("baseline output")).toBeTruthy();
    expect(within(screen.getByRole("region", { name: "Candidate example detail" })).getByText("candidate output")).toBeTruthy();
  });

  it("holds top-level surface and run navigation when a review flush fails", async () => {
    const annotatedDetail = {
      ...detail,
      annotation: { note: null, tags: [], verdict: "should_be_parseable" as const },
    };
    const api = fakeApi({
      getExample: vi.fn().mockResolvedValue(annotatedDetail),
      putAnnotation: vi.fn().mockRejectedValue(new Error("database is locked")),
    });
    render(<PreprocessingViewer api={api} />);
    await screen.findByRole("heading", { name: "Trace every stage back to examples" });
    fireEvent.click(screen.getByRole("button", { name: "Review" }));
    await screen.findByRole("heading", { name: "sample-1" });

    fireEvent.change(screen.getByLabelText("Note"), { target: { value: "keep this draft" } });
    fireEvent.click(screen.getByRole("button", { name: "Waterfall" }));
    expect(await screen.findByText("Save failed")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Triage terminal preprocessing failures" })).toBeTruthy();

    const runSelect = screen.getByLabelText("Active run") as HTMLSelectElement;
    fireEvent.change(runSelect, { target: { value: "candidate" } });
    await waitFor(() => expect(runSelect.value).toBe("baseline"));
    expect((screen.getByLabelText("Note") as HTMLTextAreaElement).value).toBe("keep this draft");
  });

  it("holds App-mounted internal example navigation when a review flush fails", async () => {
    const firstDetail = {
      ...detail,
      annotation: { note: null, tags: [], verdict: "should_be_parseable" as const },
    };
    const secondDetail = { ...detail, decoder_output_sha256: "second-output", sample_id: "sample-2" };
    const api = fakeApi({
      getExample: vi.fn(async (_runId: string, sampleId: string) => sampleId === "sample-2" ? secondDetail : firstDetail),
      getExamples: vi.fn().mockResolvedValue({
        items: [
          { annotation_verdict: "should_be_parseable", context: {}, outcome: detail.outcome, raw_preview: "first", sample_id: "sample-1" },
          { annotation_verdict: null, context: {}, outcome: detail.outcome, raw_preview: "second", sample_id: "sample-2" },
        ],
        limit: 30,
        offset: 0,
        total: 2,
      }),
      putAnnotation: vi.fn().mockRejectedValue(new Error("database is locked")),
    });
    render(<PreprocessingViewer api={api} />);
    await screen.findByRole("heading", { name: "Trace every stage back to examples" });
    fireEvent.click(screen.getByRole("button", { name: "Review" }));
    await screen.findByRole("heading", { name: "sample-1" });

    fireEvent.change(screen.getByLabelText("Note"), { target: { value: "keep internal draft" } });
    fireEvent.click(screen.getByRole("button", { name: /sample-2/ }));

    expect(await screen.findByText("Save failed")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "sample-1" })).toBeTruthy();
    expect((screen.getByLabelText("Note") as HTMLTextAreaElement).value).toBe("keep internal draft");
  });
});
