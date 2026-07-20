import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@dr-code/viewer", () => ({
  StatusBadge: ({ children }: { children: ReactNode }) => <span>{children}</span>,
}));

import { FailureExamplesLoader, type Example, type FailureBrowserGroup, type FailureIndexEntry } from "../src/data";
import { FailureBrowser } from "../src/failure-browser";

afterEach(cleanup);

function entry(sampleId: string, detailShard = `details/${sampleId}.json`): FailureIndexEntry {
  return {
    context: { source_kind: "fixture" },
    detail_shard: detailShard,
    failed_step: "compile_candidates",
    outcome: "no_compilable_candidate",
    raw_character_count: 12,
    rejection_reasons: ["compile_failed"],
    sample_id: sampleId,
  };
}

function example(sampleId: string): Example {
  return {
    candidates: [],
    categories: [],
    context: { source_kind: "fixture" },
    facts: [],
    final_candidate_count: 0,
    outcome: "no_compilable_candidate",
    raw_decoder_output: `response for ${sampleId}`,
    rejections: [{ details_json: "{}", reason_code: null, step_name: "compile_candidates" }],
    sample_id: sampleId,
  };
}

function indexPayload(group: FailureBrowserGroup, entries: FailureIndexEntry[]) {
  return {
    schema_version: 1,
    failure_code: group.failure_code,
    failed_step: group.failed_step,
    count: entries.length,
    entries,
  };
}

function response(payload: unknown, ok = true, status = 200) {
  return { json: async () => payload, ok, status };
}

function renderBrowser(
  groups: FailureBrowserGroup[],
  loader: FailureExamplesLoader,
) {
  return render(
    <FailureBrowser
      browser={{
        artifact_id: "fixture-artifact",
        groups,
        schema_version: 1,
        total_count: groups.reduce((sum, group) => sum + group.count, 0),
      }}
      loader={loader}
      renderExample={(value) => <div data-testid="failure-detail">{value.sample_id}</div>}
    />,
  );
}

describe("FailureBrowser", () => {
  it("does not request artifacts until a tuple-identifying cause card is activated", async () => {
    const compileGroup: FailureBrowserGroup = {
      count: 1,
      failed_step: "compile_candidates",
      failure_code: "terminal_failure",
      index_path: "indexes/compile.json",
    };
    const functionGroup: FailureBrowserGroup = {
      count: 1,
      failed_step: "find_top_level_function",
      failure_code: "terminal_failure",
      index_path: "indexes/function.json",
    };
    const functionEntry = entry("sample-function", "details/function.json");
    const payloads: Record<string, unknown> = {
      "/app/data/failure-examples/indexes/function.json": indexPayload(functionGroup, [functionEntry]),
      "/app/data/failure-examples/details/function.json": {
        schema_version: 1,
        failure_code: "terminal_failure",
        examples: [example("sample-function")],
      },
    };
    const fetchArtifact = vi.fn(async (path: string) => response(payloads[path]));
    const loader = new FailureExamplesLoader(fetchArtifact, "/app/");

    renderBrowser([compileGroup, functionGroup], loader);

    expect(fetchArtifact).not.toHaveBeenCalled();
    expect(screen.getByText("Select a terminal cause to load its failure index.")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", {
      name: /terminal failure · find top level function 1 examples/i,
    }));

    expect((await screen.findByTestId("failure-detail")).textContent).toBe("sample-function");
    expect(fetchArtifact).toHaveBeenCalledTimes(2);
    expect(fetchArtifact).not.toHaveBeenCalledWith(
      "/app/data/failure-examples/indexes/compile.json",
    );
    expect(screen.getByRole("button", {
      name: /terminal failure · find top level function 1 examples/i,
    }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByRole("button", {
      name: /terminal failure · compile candidates 1 examples/i,
    }).getAttribute("aria-pressed")).toBe("false");
  });

  it("never paints a stale detail response after the active sample changes", async () => {
    const group: FailureBrowserGroup = {
      count: 2,
      failed_step: "compile_candidates",
      failure_code: "compile_failed",
      index_path: "index.json",
    };
    const firstEntry = entry("sample-first");
    const secondEntry = entry("sample-second");
    let resolveFirst: ((value: ReturnType<typeof response>) => void) | undefined;
    const firstDetail = new Promise<ReturnType<typeof response>>((resolve) => {
      resolveFirst = resolve;
    });
    const fetchArtifact = vi.fn((path: string) => {
      if (path.endsWith("index.json")) {
        return Promise.resolve(response(indexPayload(group, [firstEntry, secondEntry])));
      }
      if (path.endsWith("sample-first.json")) return firstDetail;
      return Promise.resolve(response({
        schema_version: 1,
        failure_code: "compile_failed",
        examples: [example("sample-second")],
      }));
    });
    const loader = new FailureExamplesLoader(fetchArtifact, "/");

    renderBrowser([group], loader);
    fireEvent.click(screen.getByRole("button", { name: /compile failed · compile candidates/i }));
    await screen.findByRole("button", { name: /sample-second/i });
    fireEvent.click(screen.getByRole("button", { name: /sample-second/i }));

    expect((await screen.findByTestId("failure-detail")).textContent).toBe("sample-second");

    await act(async () => {
      resolveFirst?.(response({
        schema_version: 1,
        failure_code: "compile_failed",
        examples: [example("sample-first")],
      }));
      await firstDetail;
    });
    await waitFor(() => {
      expect(screen.getByTestId("failure-detail").textContent).toBe("sample-second");
    });
  });

  it("shows an index error and retries the evicted request", async () => {
    const group: FailureBrowserGroup = {
      count: 1,
      failed_step: "compile_candidates",
      failure_code: "compile_failed",
      index_path: "index.json",
    };
    const onlyEntry = entry("sample-recovered");
    const fetchArtifact = vi
      .fn()
      .mockResolvedValueOnce(response({}, false, 503))
      .mockResolvedValueOnce(response(indexPayload(group, [onlyEntry])))
      .mockResolvedValueOnce(response({
        schema_version: 1,
        failure_code: "compile_failed",
        examples: [example("sample-recovered")],
      }));
    const loader = new FailureExamplesLoader(fetchArtifact, "/");

    renderBrowser([group], loader);
    fireEvent.click(screen.getByRole("button", { name: /compile failed · compile candidates/i }));

    expect(await screen.findByText("Could not load this failure group.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Retry failure group" }));

    expect((await screen.findByTestId("failure-detail")).textContent).toBe("sample-recovered");
    expect(fetchArtifact).toHaveBeenCalledTimes(3);
  });

  it("shows a detail error and retries only that detail shard", async () => {
    const group: FailureBrowserGroup = {
      count: 1,
      failed_step: "compile_candidates",
      failure_code: "compile_failed",
      index_path: "index.json",
    };
    const onlyEntry = entry("sample-detail-recovered");
    const fetchArtifact = vi
      .fn()
      .mockResolvedValueOnce(response(indexPayload(group, [onlyEntry])))
      .mockResolvedValueOnce(response({}, false, 503))
      .mockResolvedValueOnce(response({
        schema_version: 1,
        failure_code: "compile_failed",
        examples: [example("sample-detail-recovered")],
      }));
    const loader = new FailureExamplesLoader(fetchArtifact, "/");

    renderBrowser([group], loader);
    fireEvent.click(screen.getByRole("button", { name: /compile failed · compile candidates/i }));

    expect(await screen.findByText("Could not load this example.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Retry failure detail" }));

    expect((await screen.findByTestId("failure-detail")).textContent).toBe("sample-detail-recovered");
    expect(fetchArtifact).toHaveBeenCalledTimes(3);
    expect(fetchArtifact).toHaveBeenNthCalledWith(1, "/data/failure-examples/index.json");
  });
});
