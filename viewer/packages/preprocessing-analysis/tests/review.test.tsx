import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@dr-code/viewer", () => ({
  CodeBlock: ({ code }: { code: string }) => <pre>{code}</pre>,
  CodeDiff: () => <div>code diff</div>,
  StatusBadge: ({ children }: { children: ReactNode }) => <span>{children}</span>,
}));

import { Review } from "../src/review";
import { detail, fakeApi } from "./fixtures";

afterEach(cleanup);

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((promiseResolve) => { resolve = promiseResolve; });
  return { promise, resolve };
}

describe("Review", () => {
  it("uses the full failure tuple and saves verdict, note, tags, and clear actions", async () => {
    const api = fakeApi();
    const onTagCreated = vi.fn();
    render(<Review api={api} onTagCreated={onTagCreated} runId="baseline" tags={[]} />);

    expect(await screen.findByRole("heading", { name: "sample-1" })).toBeTruthy();
    expect(screen.getByText("unreviewed")).toBeTruthy();
    expect(api.getExamples).toHaveBeenCalledWith("baseline", expect.objectContaining({
      cause: "syntax error",
      failed_step: "compile",
      failure_code: "syntax_error",
      limit: 30,
      offset: 0,
    }));
    fireEvent.click(screen.getByRole("radio", { name: /Should be parseable/ }));
    await waitFor(() => expect(api.putAnnotation).toHaveBeenCalledWith(
      expect.objectContaining({ decoder_output_sha256: "output-sha", sample_id: "sample-1" }),
      { note: "", tag_ids: [], verdict: "should_be_parseable" },
    ));
    expect(await screen.findByText("Saved")).toBeTruthy();
    expect(screen.getByText("should be parseable")).toBeTruthy();

    fireEvent.change(screen.getByLabelText("Note"), { target: { value: "Recover the fenced function" } });
    expect(screen.getByText("Unsaved changes")).toBeTruthy();
    await waitFor(() => expect(api.putAnnotation).toHaveBeenLastCalledWith(
      expect.anything(),
      expect.objectContaining({ note: "Recover the fenced function" }),
    ));
    expect(await screen.findByText("Saved")).toBeTruthy();

    fireEvent.change(screen.getByLabelText("Create tag"), { target: { value: "markdown fence" } });
    fireEvent.click(screen.getByRole("button", { name: "Create and select" }));
    await waitFor(() => expect(api.createTag).toHaveBeenCalledWith("markdown fence"));
    expect(onTagCreated).toHaveBeenCalledWith({ name: "markdown fence", tag_id: "tag-1" });

    fireEvent.click(screen.getByRole("button", { name: "Clear annotation" }));
    await waitFor(() => expect(api.deleteAnnotation).toHaveBeenCalledWith(expect.objectContaining({ sample_id: detail.sample_id })));
  });

  it("distinguishes nonempty, null, and empty failure causes", async () => {
    const api = fakeApi();
    render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={[]} />);
    await screen.findByRole("heading", { name: "sample-1" });

    fireEvent.click(screen.getByRole("button", { name: /Literal response/ }));
    await waitFor(() => expect(api.getExamples).toHaveBeenCalledWith("baseline", expect.objectContaining({
      cause_is_null: true,
      failed_step: "compile",
      failure_code: "syntax_error",
    })));
    fireEvent.click(screen.getByRole("button", { name: /Empty cause/ }));
    await waitFor(() => expect(api.getExamples).toHaveBeenCalledWith("baseline", expect.objectContaining({
      cause: "",
      failed_step: "compile",
      failure_code: "syntax_error",
    })));
  });

  it("does not expose annotation controls for an output without a digest", async () => {
    const api = fakeApi({ getExample: vi.fn().mockResolvedValue({ ...detail, decoder_output_sha256: null }) });
    render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={[]} />);

    const verdict = await screen.findByRole("radio", { name: /Should be parseable/ });
    expect((verdict as HTMLInputElement).disabled).toBe(true);
  });

  it("keeps failed annotation saves visible", async () => {
    const api = fakeApi({ putAnnotation: vi.fn().mockRejectedValue(new Error("database is locked")) });
    render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={[]} />);

    fireEvent.click(await screen.findByRole("radio", { name: /Expected no code/ }));
    expect(await screen.findByText("Save failed")).toBeTruthy();
    expect(screen.getByRole("alert").textContent).toContain("database is locked");
  });

  it("serializes rapid full-state saves and coalesces them to the latest draft", async () => {
    const first = deferred<NonNullable<typeof detail.annotation>>();
    const second = deferred<NonNullable<typeof detail.annotation>>();
    const putAnnotation = vi.fn()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const tag = { name: "markdown fence", tag_id: "tag-1" };
    const api = fakeApi({ putAnnotation });
    render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={[tag]} />);

    fireEvent.click(await screen.findByRole("radio", { name: /Should be parseable/ }));
    fireEvent.click(screen.getByRole("radio", { name: /Expected no code/ }));
    fireEvent.change(screen.getByLabelText("Note"), { target: { value: "latest note" } });
    fireEvent.click(screen.getByRole("checkbox", { name: "markdown fence" }));
    expect(putAnnotation).toHaveBeenCalledTimes(1);

    await act(async () => {
      first.resolve({ note: null, tags: [], verdict: "should_be_parseable" });
      await first.promise;
    });
    await waitFor(() => expect(putAnnotation).toHaveBeenCalledTimes(2));
    expect(putAnnotation).toHaveBeenLastCalledWith(expect.anything(), {
      note: "latest note",
      tag_ids: ["tag-1"],
      verdict: "expected_no_code",
    });

    await act(async () => {
      second.resolve({ note: "latest note", tags: [tag], verdict: "expected_no_code" });
      await second.promise;
    });
    expect(await screen.findByText("Saved")).toBeTruthy();
  });

  it("flushes a dirty note on navigation without applying its completion to the new example", async () => {
    const pending = deferred<NonNullable<typeof detail.annotation>>();
    const firstDetail = {
      ...detail,
      annotation: { note: null, tags: [], verdict: "should_be_parseable" as const },
    };
    const secondDetail = { ...detail, decoder_output_sha256: "second-output", sample_id: "sample-2" };
    const api = fakeApi({
      getExample: vi.fn(async (_runId: string, sampleId: string) => sampleId === "sample-2" ? secondDetail : firstDetail),
      getExamples: vi.fn().mockResolvedValue({
        items: [
          { annotation_verdict: null, context: {}, outcome: detail.outcome, raw_preview: "first", sample_id: "sample-1" },
          { annotation_verdict: null, context: {}, outcome: detail.outcome, raw_preview: "second", sample_id: "sample-2" },
        ],
        limit: 30,
        offset: 0,
        total: 2,
      }),
      putAnnotation: vi.fn().mockReturnValue(pending.promise),
    });
    render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={[]} />);

    await screen.findByRole("heading", { name: "sample-1" });
    fireEvent.change(screen.getByLabelText("Note"), { target: { value: "flush before navigation" } });
    expect(screen.getByText("Unsaved changes")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /sample-2/ }));
    await waitFor(() => expect(api.putAnnotation).toHaveBeenCalledWith(
      expect.objectContaining({ sample_id: "sample-1" }),
      { note: "flush before navigation", tag_ids: [], verdict: "should_be_parseable" },
    ));
    expect(screen.getByRole("heading", { name: "sample-1" })).toBeTruthy();
    await act(async () => {
      pending.resolve({ note: null, tags: [], verdict: "should_be_parseable" });
      await pending.promise;
    });

    await screen.findByRole("heading", { name: "sample-2" });
    expect((screen.getByRole("radio", { name: /Should be parseable/ }) as HTMLInputElement).checked).toBe(false);
    expect(screen.getByText("should be parseable")).toBeTruthy();
  });

  it("holds navigation when a dirty-note flush fails and keeps the draft recoverable", async () => {
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
    render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={[]} />);

    await screen.findByRole("heading", { name: "sample-1" });
    fireEvent.change(screen.getByLabelText("Note"), { target: { value: "recoverable draft" } });
    fireEvent.click(screen.getByRole("button", { name: /sample-2/ }));

    expect(await screen.findByText("Save failed")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "sample-1" })).toBeTruthy();
    expect((screen.getByLabelText("Note") as HTMLTextAreaElement).value).toBe("recoverable draft");
  });

  it("resets pagination and search when the active run changes", async () => {
    const getExamples = vi.fn().mockResolvedValue({
      items: [{ annotation_verdict: null, context: {}, outcome: detail.outcome, raw_preview: "first", sample_id: "sample-1" }],
      limit: 30,
      offset: 0,
      total: 60,
    });
    const api = fakeApi({ getExamples });
    const view = render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={[]} />);
    await screen.findByRole("heading", { name: "sample-1" });

    fireEvent.change(screen.getByLabelText("Search this group"), { target: { value: "needle" } });
    await waitFor(() => expect(getExamples).toHaveBeenCalledWith("baseline", expect.objectContaining({ search: "needle" })));
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() => expect(getExamples).toHaveBeenCalledWith("baseline", expect.objectContaining({ offset: 30 })));

    view.rerender(<Review api={api} onTagCreated={vi.fn()} runId="candidate" tags={[]} />);
    await waitFor(() => expect(getExamples).toHaveBeenCalledWith("candidate", expect.objectContaining({
      offset: 0,
      search: "",
    })));
    expect((screen.getByLabelText("Search this group") as HTMLInputElement).value).toBe("");
  });
});
