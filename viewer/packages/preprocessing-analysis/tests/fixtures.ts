import { vi } from "vitest";

import type {
  Annotation,
  CompareResponse,
  ExampleDetail,
  ExamplesResponse,
  FailuresResponse,
  PreprocessingApi,
  RunSummary,
  Tag,
  TaskAnnotation,
  WaterfallResponse,
} from "../src/api";

export const baselineRun: RunSummary = {
  corpus_sha256: "corpus-baseline-0123456789",
  dataset_id: "evalplus/humanevalplus",
  definition_id: "preprocessing",
  has_evaluation: false,
  label: "Baseline",
  manifest_sha256: "manifest-baseline",
  run_id: "baseline",
  semantic_coordinates: { definition_version: "1" },
};

export const candidateRun: RunSummary = {
  ...baselineRun,
  label: "Candidate",
  manifest_sha256: "manifest-candidate",
  run_id: "candidate",
  semantic_coordinates: { definition_version: "2" },
};

export const detail: ExampleDetail = {
  annotation: null,
  candidates: [],
  cause: "syntax error",
  context: { source_kind: "fixture" },
  corpus_sha256: baselineRun.corpus_sha256,
  dataset_id: baselineRun.dataset_id,
  decoder_output_sha256: "output-sha",
  failed_step: "compile",
  failure_code: "syntax_error",
  facts: [{ facts_json: "{}", step_name: "extract" }],
  outcome: "no_compilable_candidate",
  raw_decoder_output: "```python\ndef broken(:\n```",
  rejections: [{ details_json: "{}", reason_code: "syntax_error", step_name: "compile" }],
  sample_id: "sample-1",
  task_identity: "a".repeat(64),
};

export const examples: ExamplesResponse = {
  items: [{
    annotation_verdict: null,
    context: { task_id: "task-1" },
    outcome: detail.outcome,
    raw_preview: detail.raw_decoder_output,
    sample_id: detail.sample_id,
  }],
  limit: 30,
  offset: 0,
  total: 1,
};

export const waterfall: WaterfallResponse = {
  run_id: baselineRun.run_id,
  stages: [
    { count: 10, denominator_count: 10, id: "source", label: "Source samples", rate: 1, unit: "sample" },
    { count: 7, denominator_count: 10, id: "output_nonblank", label: "Nonblank output", rate: .7, unit: "sample" },
  ],
};

export const failures: FailuresResponse = {
  groups: [
    { cause: "syntax error", count: 1, failed_step: "compile", failure_code: "syntax_error", id: "syntax_error:compile:syntax error", label: "Compilation failed", reason_code: "syntax error" },
    { cause: null, count: 2, failed_step: "compile", failure_code: "syntax_error", id: "syntax_error:compile:null", label: "Literal response", reason_code: null },
    { cause: "", count: 1, failed_step: "compile", failure_code: "syntax_error", id: "syntax_error:compile:empty", label: "Empty cause", reason_code: "" },
  ],
  run_id: baselineRun.run_id,
  total_count: 4,
};

export const comparison: CompareResponse = {
  baseline_run_id: baselineRun.run_id,
  candidate_run_id: candidateRun.run_id,
  compatible: true,
  incompatibility_reason: null,
  stages: [{
    baseline_count: 7,
    baseline_denominator_count: 10,
    baseline_rate: .7,
    candidate_count: 8,
    candidate_denominator_count: 10,
    candidate_rate: .8,
    count_delta: 1,
    label: "Candidates extracted",
    rate_delta: .1,
    id: "has_extracted_candidate",
    unit: "sample",
  }],
  transitions: [{
    baseline_outcome: "no_compilable_candidate",
    candidate_outcome: "function_candidates_extracted",
    count: 1,
    id: "no_compilable_candidate → function_candidates_extracted",
  }],
};

export function fakeApi(overrides: Partial<PreprocessingApi> = {}): PreprocessingApi {
  const saved: Annotation = { note: null, tags: [], verdict: "should_be_parseable" };
  const savedTask: TaskAnnotation = {
    category: null,
    identity: {
      dataset_id: baselineRun.dataset_id,
      task_id: "HumanEval/1",
      task_identity: "a".repeat(64),
    },
    note: null,
    origin: "human",
    provenance: null,
    tags: [],
  };
  const tag: Tag = { name: "markdown fence", tag_id: "tag-1" };
  return {
    compare: vi.fn().mockResolvedValue(comparison),
    createTag: vi.fn().mockResolvedValue(tag),
    deleteAnnotation: vi.fn().mockResolvedValue(undefined),
    deleteTaskAnnotation: vi.fn().mockResolvedValue(undefined),
    getExample: vi.fn().mockResolvedValue(detail),
    getExamples: vi.fn().mockResolvedValue(examples),
    getFailures: vi.fn().mockResolvedValue(failures),
    getReviewExamples: vi.fn().mockResolvedValue({
      items: [detail],
      limit: 10,
      offset: 0,
      total: 1,
    }),
    getRuns: vi.fn().mockResolvedValue([baselineRun, candidateRun]),
    getTags: vi.fn().mockResolvedValue([]),
    getTaskAnnotation: vi.fn().mockResolvedValue(null),
    getWaterfall: vi.fn().mockResolvedValue(waterfall),
    putAnnotation: vi.fn().mockResolvedValue(saved),
    putTaskAnnotation: vi.fn().mockResolvedValue(savedTask),
    ...overrides,
  };
}
