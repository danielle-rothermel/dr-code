import { describe, expect, it, vi } from "vitest";

import { ApiError, HttpPreprocessingApi } from "../src/api";

function response(payload: unknown, status = 200) {
  return {
    json: vi.fn().mockResolvedValue(payload),
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Conflict",
  };
}

describe("HttpPreprocessingApi", () => {
  it("calls the default browser fetch transport with the global receiver", async () => {
    const browserFetch = vi.fn(function (this: unknown) {
      expect(this).toBe(globalThis);
      return Promise.resolve(response([]));
    });
    vi.stubGlobal("fetch", browserFetch);

    try {
      await expect(new HttpPreprocessingApi().getRuns()).resolves.toEqual([]);
      expect(browserFetch).toHaveBeenCalledWith("/api/runs", undefined);
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("encodes exact example filters and annotation identities", async () => {
    const transport = vi.fn().mockResolvedValue(response({ items: [], limit: 50, offset: 0, total: 0 }));
    const api = new HttpPreprocessingApi(transport, "/viewer");

    await api.getExamples("run/id", {
      baseline_outcome: "no code",
      candidate_outcome: "function",
      compare_run_id: "candidate/id",
      cause: "syntax error",
      failed_step: "compile candidates",
      failure_code: "syntax/error",
      limit: 50,
      offset: 100,
    });

    expect(transport).toHaveBeenCalledWith(
      "/viewer/api/runs/run%2Fid/examples?baseline_outcome=no+code&candidate_outcome=function&compare_run_id=candidate%2Fid&cause=syntax+error&failed_step=compile+candidates&failure_code=syntax%2Ferror&limit=50&offset=100",
      undefined,
    );

    await api.getExamples("run/id", { cause_is_null: true, failure_code: "syntax/error" });
    expect(transport).toHaveBeenLastCalledWith(
      "/viewer/api/runs/run%2Fid/examples?cause_is_null=true&failure_code=syntax%2Ferror",
      undefined,
    );

    await api.getExamples("run/id", { cause: "", failure_code: "syntax/error" });
    expect(transport).toHaveBeenLastCalledWith(
      "/viewer/api/runs/run%2Fid/examples?cause=&failure_code=syntax%2Ferror",
      undefined,
    );

    await api.getReviewExamples("run/id", {
      cause_is_null: true,
      failed_step: "compile candidates",
      failure_code: "syntax/error",
      limit: 25,
      offset: 50,
      search: "task/id",
    });
    expect(transport).toHaveBeenLastCalledWith(
      "/viewer/api/runs/run%2Fid/review-examples?cause_is_null=true&failed_step=compile+candidates&failure_code=syntax%2Ferror&limit=25&offset=50&search=task%2Fid",
      undefined,
    );
  });

  it("uses FastAPI detail messages for rejected comparisons", async () => {
    const api = new HttpPreprocessingApi(vi.fn().mockResolvedValue(response({ detail: "Corpus fingerprints differ" }, 409)));

    await expect(api.compare("before", "after")).rejects.toEqual(
      new ApiError("Corpus fingerprints differ", 409),
    );
  });
});
