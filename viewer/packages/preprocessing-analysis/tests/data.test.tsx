import { describe, expect, it, vi } from "vitest";

import {
  crosstabRows,
  FailureExamplesLoader,
  filterEvaluationExamples,
  filterExamples,
  filterFailureEntries,
  paginateFailureEntries,
  type Example,
  type FailureBrowserGroup,
  type FailureIndexEntry,
  viewerData,
} from "../src/data";
import { evaluatedData } from "./evaluation-fixture";

function failureEntry(index: number, overrides: Partial<FailureIndexEntry> = {}): FailureIndexEntry {
  return {
    context: { source_kind: index === 51 ? "searchable-source" : "fixture" },
    detail_shard: "details/compile-0.json",
    failed_step: "compile_candidates",
    outcome: "no_compilable_candidate",
    raw_character_count: 100 + index,
    rejection_reasons: index === 52 ? ["syntax_error"] : ["compile_failed"],
    sample_id: `sample-${String(index).padStart(2, "0")}`,
    ...overrides,
  };
}

function failureExample(sampleId: string): Example {
  return {
    candidates: [],
    categories: ["failure:compile_failed"],
    context: { source_kind: "fixture" },
    facts: [],
    final_candidate_count: 0,
    outcome: "no_compilable_candidate",
    raw_decoder_output: "def broken(:",
    rejections: [{
      details_json: "{}",
      reason_code: "compile_failed",
      step_name: "compile_candidates",
    }],
    sample_id: sampleId,
  };
}

const compileGroup: FailureBrowserGroup = {
  count: 2,
  failed_step: "compile_candidates",
  failure_code: "compile_failed",
  index_path: "indexes/compile.json",
};

describe("preprocessing analysis data selectors", () => {
  it("finds an example by source metadata and preserves its diagnostic outcome", () => {
    const matches = filterExamples(viewerData.examples, "legacy_domain_partial", "all");

    expect(matches).toHaveLength(1);
    expect(matches[0]?.outcome).toBe("decoder_output_missing");
  });

  it("combines the outcome selector with search", () => {
    const matches = filterExamples(
      viewerData.examples,
      "HumanEval/35",
      "function_candidates_extracted",
    );

    expect(matches).toHaveLength(1);
    expect(matches[0]?.final_candidate_count).toBe(1);
  });

  it("returns a count-sorted crosstab for a selected dimension and denominator", () => {
    const rows = crosstabRows("source_kind", "function_candidates_extracted", "all");

    expect(rows.length).toBeGreaterThan(1);
    expect(rows.every((row) => row.dimension === "source_kind")).toBe(true);
    expect(rows.every((row) => row.denominator === "all")).toBe(true);
    expect(rows[0]?.count).toBeGreaterThanOrEqual(rows[1]?.count ?? 0);
  });

  it("loads the joined candidate evaluation from the checked snapshot", () => {
    expect(viewerData.schema_version).toBe(2);
    expect(viewerData.candidate_evaluation?.summary.available).toBe(true);
    expect(viewerData.candidate_evaluation?.summary.candidate_membership_count).toBe(325_769);
    expect(viewerData.candidate_evaluation?.examples).toHaveLength(12);
  });

  it("searches evaluation diagnostics within a selected test outcome", () => {
    const evaluation = evaluatedData.candidate_evaluation;
    expect(evaluation).toBeDefined();

    const matches = filterEvaluationExamples(
      evaluation?.examples ?? [],
      "SandboxError",
      "infrastructure_failure",
    );

    expect(matches).toHaveLength(1);
    expect(matches[0]?.diagnostics.failure_message).toBe("runtime unavailable");
  });

  it("searches failure IDs, context, and rejection reasons and paginates by 50", () => {
    const entries = Array.from({ length: 55 }, (_, index) => failureEntry(index));

    expect(filterFailureEntries(entries, "searchable-source")).toEqual([entries[51]]);
    expect(filterFailureEntries(entries, "syntax error")).toEqual([entries[52]]);

    const firstPage = paginateFailureEntries(entries, 1);
    const secondPage = paginateFailureEntries(entries, 2);
    expect(firstPage.entries).toHaveLength(50);
    expect(firstPage).toMatchObject({ start: 1, end: 50, total: 55, page: 1, pageCount: 2 });
    expect(secondPage.entries).toHaveLength(5);
    expect(secondPage).toMatchObject({ start: 51, end: 55, total: 55, page: 2, pageCount: 2 });
  });

  it("lazily loads indexes and detail shards from the Vite base and caches each artifact", async () => {
    const entries = [failureEntry(1), failureEntry(2, { sample_id: "sample-02" })];
    const responses: Record<string, unknown> = {
      "/viewer/data/failure-examples/indexes/compile.json": {
        schema_version: 1,
        failure_code: "compile_failed",
        failed_step: "compile_candidates",
        count: 2,
        entries,
      },
      "/viewer/data/failure-examples/details/compile-0.json": {
        schema_version: 1,
        failure_code: "compile_failed",
        examples: entries.map(({ sample_id }) => failureExample(sample_id)),
      },
    };
    const fetchArtifact = vi.fn(async (path: string) => ({
      json: async () => responses[path],
      ok: path in responses,
      status: path in responses ? 200 : 404,
    }));
    const loader = new FailureExamplesLoader(fetchArtifact, "/viewer/");

    const index = await loader.loadIndex(compileGroup);
    await loader.loadIndex(compileGroup);
    const first = await loader.loadDetail(index.entries[0]!, compileGroup.failure_code);
    const second = await loader.loadDetail(index.entries[1]!, compileGroup.failure_code);

    expect(first.sample_id).toBe("sample-01");
    expect(second.sample_id).toBe("sample-02");
    expect(fetchArtifact).toHaveBeenCalledTimes(2);
    expect(fetchArtifact).toHaveBeenNthCalledWith(
      1,
      "/viewer/data/failure-examples/indexes/compile.json",
    );
    expect(fetchArtifact).toHaveBeenNthCalledWith(
      2,
      "/viewer/data/failure-examples/details/compile-0.json",
    );
  });

  it("evicts failed requests so an explicit retry can load the artifact", async () => {
    const indexPayload = {
      schema_version: 1,
      failure_code: "compile_failed",
      failed_step: "compile_candidates",
      count: 2,
      entries: [failureEntry(1), failureEntry(2)],
    };
    const fetchArtifact = vi
      .fn()
      .mockResolvedValueOnce({ json: async () => ({}), ok: false, status: 503 })
      .mockResolvedValueOnce({ json: async () => indexPayload, ok: true, status: 200 });
    const loader = new FailureExamplesLoader(fetchArtifact, "/");

    await expect(loader.loadIndex(compileGroup)).rejects.toThrow("HTTP 503");
    await expect(loader.loadIndex(compileGroup)).resolves.toMatchObject({ count: 2 });
    expect(fetchArtifact).toHaveBeenCalledTimes(2);
  });
});
