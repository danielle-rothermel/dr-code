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
      "/viewer/api/examples?run_id=run%2Fid&baseline_outcome=no+code&candidate_outcome=function&compare_run_id=candidate%2Fid&cause=syntax+error&failed_step=compile+candidates&failure_code=syntax%2Ferror&limit=50&offset=100",
      undefined,
    );

    await api.getExamples("run/id", { cause_is_null: true, failure_code: "syntax/error" });
    expect(transport).toHaveBeenLastCalledWith(
      "/viewer/api/examples?run_id=run%2Fid&cause_is_null=true&failure_code=syntax%2Ferror",
      undefined,
    );

    await api.getExamples("run/id", { cause: "", failure_code: "syntax/error" });
    expect(transport).toHaveBeenLastCalledWith(
      "/viewer/api/examples?run_id=run%2Fid&cause=&failure_code=syntax%2Ferror",
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
      "/viewer/api/review-examples?run_id=run%2Fid&cause_is_null=true&failed_step=compile+candidates&failure_code=syntax%2Ferror&limit=25&offset=50&search=task%2Fid",
      undefined,
    );

    await api.getExample("run/id", "HumanEval/32");
    expect(transport).toHaveBeenLastCalledWith(
      "/viewer/api/example?run_id=run%2Fid&sample_id=HumanEval%2F32",
      undefined,
    );

    const identity = {
      corpus_sha256: "a".repeat(64),
      decoder_output_sha256: "b".repeat(64),
      sample_id: "HumanEval/32",
    };
    await api.putAnnotation(identity, { note: "", tag_ids: [], verdict: null });
    expect(transport).toHaveBeenLastCalledWith(
      `/viewer/api/annotations/${"a".repeat(64)}/${"b".repeat(64)}?sample_id=HumanEval%2F32`,
      {
        body: JSON.stringify({ note: "", tag_ids: [], verdict: null }),
        headers: { "Content-Type": "application/json" },
        method: "PUT",
      },
    );
    await api.deleteAnnotation(identity);
    expect(transport).toHaveBeenLastCalledWith(
      `/viewer/api/annotations/${"a".repeat(64)}/${"b".repeat(64)}?sample_id=HumanEval%2F32`,
      { method: "DELETE" },
    );
  });

  it("uses FastAPI detail messages for rejected comparisons", async () => {
    const api = new HttpPreprocessingApi(vi.fn().mockResolvedValue(response({ detail: "Corpus fingerprints differ" }, 409)));

    await expect(api.compare("before", "after")).rejects.toEqual(
      new ApiError("Corpus fingerprints differ", 409),
    );
  });

  it("uses the nested task-annotation API without exposing a machine write", async () => {
    const task = {
      category: "algorithm",
      identity: {
        dataset_id: "evalplus/humanevalplus",
        task_id: "HumanEval/0",
        task_identity: "a".repeat(64),
      },
      note: null,
      origin: "human" as const,
      provenance: null,
      tags: [],
    };
    const transport = vi.fn()
      .mockResolvedValueOnce(response(task))
      .mockResolvedValueOnce(response(task))
      .mockResolvedValueOnce(response(undefined, 204))
      .mockResolvedValueOnce(response({ detail: "Not Found" }, 404));
    const api = new HttpPreprocessingApi(transport, "/viewer");
    const identity = {
      dataset_id: "evalplus/humanevalplus",
      task_id: "HumanEval/0",
      task_identity: "a".repeat(64),
    };

    await expect(api.getTaskAnnotation(identity)).resolves.toEqual(task);
    await api.putTaskAnnotation(identity, {
      category: "algorithm",
      note: null,
      tag_ids: [],
    });
    await api.deleteTaskAnnotation(identity);
    await expect(api.getTaskAnnotation(identity)).resolves.toBeNull();

    const path = (
      "/viewer/api/task-annotations"
      + "?dataset_id=evalplus%2Fhumanevalplus&task_id=HumanEval%2F0"
      + `&task_identity=${"a".repeat(64)}`
    );
    expect(transport).toHaveBeenNthCalledWith(1, path, undefined);
    expect(transport).toHaveBeenNthCalledWith(2, path, {
      body: JSON.stringify({
        category: "algorithm",
        note: null,
        tag_ids: [],
      }),
      headers: { "Content-Type": "application/json" },
      method: "PUT",
    });
    expect(transport).toHaveBeenNthCalledWith(3, path, {
      method: "DELETE",
    });
  });
});
