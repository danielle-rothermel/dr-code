export type Unit = "sample" | "candidate";
export type Verdict = "should_be_parseable" | "expected_no_code";
export type JsonValue =
  | boolean
  | number
  | string
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export interface RunSummary {
  corpus_sha256: string;
  dataset_id: string;
  definition_id: string;
  has_evaluation: boolean;
  label: string;
  manifest_sha256: string;
  run_id: string;
  semantic_coordinates: Record<string, string | number | boolean | null>;
}

export interface WaterfallStage {
  count: number;
  denominator_count: number;
  description?: string | null;
  failure_count?: number | null;
  failure_label?: string | null;
  id: string;
  label: string;
  rate: number | null;
  unit: Unit;
}

export interface WaterfallResponse {
  run_id: string;
  stages: WaterfallStage[];
}

export interface FailureGroup {
  cause: string | null;
  count: number;
  failed_step: string;
  failure_code: string;
  id: string;
  label: string;
  reason_code: string | null;
}

export interface FailuresResponse {
  groups: FailureGroup[];
  run_id: string;
  total_count: number;
}

export interface ExampleSummary {
  annotation_verdict: Verdict | null;
  context: Record<string, string | number | boolean | null>;
  outcome: string;
  raw_preview: string | null;
  sample_id: string;
}

export interface ExamplesResponse {
  items: ExampleSummary[];
  limit: number;
  offset: number;
  total: number;
}

export interface Candidate {
  candidate_id: string;
  candidate_index: number;
  cleaned_source: string;
  compile_warnings: string[];
  origins: CandidateOrigin[];
  top_level_function_names: string[];
}

export interface CandidateOrigin {
  path: ExtractionOperation[];
}

export interface ExtractionOperation {
  details: Record<string, JsonValue>;
  kind: string;
}

export interface DiagnosticFact {
  facts_json: string;
  step_name: string;
}

export interface Rejection {
  details_json: string;
  reason_code: string | null;
  step_name: string;
}

export interface Tag {
  name: string;
  tag_id: string;
}

export interface Annotation {
  note: string | null;
  tags: Tag[];
  verdict: Verdict | null;
}

export interface ExampleDetail {
  annotation: Annotation | null;
  candidates: Candidate[];
  cause: string | null;
  context: Record<string, string | number | boolean | null>;
  corpus_sha256: string;
  dataset_id: string | null;
  decoder_output_sha256: string | null;
  failed_step: string | null;
  failure_code: string | null;
  facts: DiagnosticFact[];
  outcome: string;
  raw_decoder_output: string | null;
  rejections: Rejection[];
  sample_id: string;
  task_identity: string | null;
}

export interface ComparisonStage {
  baseline_count: number;
  baseline_denominator_count: number;
  baseline_rate: number | null;
  candidate_count: number;
  candidate_denominator_count: number;
  candidate_rate: number | null;
  count_delta: number;
  label: string;
  rate_delta: number | null;
  id: string;
  unit: Unit;
}

export interface OutcomeTransition {
  baseline_outcome: string;
  candidate_outcome: string;
  count: number;
  id: string;
}

export interface CompareResponse {
  baseline_run_id: string;
  candidate_run_id: string;
  compatible: boolean;
  incompatibility_reason: string | null;
  stages: ComparisonStage[];
  transitions: OutcomeTransition[];
}

export interface ExampleQuery {
  baseline_outcome?: string;
  candidate_outcome?: string;
  compare_run_id?: string;
  cause?: string;
  cause_is_null?: boolean;
  failure_code?: string;
  failed_step?: string;
  limit?: number;
  offset?: number;
  search?: string;
  stage_id?: string;
}

export interface AnnotationInput {
  note: string;
  tag_ids: string[];
  verdict: Verdict | null;
}

export interface ReviewExamplesQuery {
  cause?: string;
  cause_is_null?: boolean;
  failed_step: string;
  failure_code: string;
  limit: number;
  offset: number;
  search?: string;
}

export interface ReviewExamplesResponse {
  items: ExampleDetail[];
  limit: number;
  offset: number;
  total: number;
}

export interface AnnotationIdentity {
  corpus_sha256: string;
  decoder_output_sha256: string;
  sample_id: string;
}

export type TaskAnnotationOrigin = "human" | "machine";

export interface TaskAnnotationIdentity {
  dataset_id: string;
  task_id: string;
  task_identity: string;
}

export interface TaskAnnotationProvenance {
  agreement: number | null;
  extra: Record<string, JsonValue>;
  model: string | null;
  repeats: number | null;
  taxonomy_version: string | null;
}

export interface TaskAnnotation {
  category: string | null;
  identity: TaskAnnotationIdentity;
  note: string | null;
  origin: TaskAnnotationOrigin;
  provenance: TaskAnnotationProvenance | null;
  tags: Tag[];
}

export interface TaskAnnotationInput {
  category: string | null;
  note: string | null;
  tag_ids: string[];
}

export const TASK_CATEGORY_MAX_LENGTH = 256;
export const TASK_NOTE_MAX_LENGTH = 10_000;
export const TASK_TAG_IDS_MAX_ITEMS = 100;

export interface PreprocessingApi {
  compare(baselineRunId: string, candidateRunId: string): Promise<CompareResponse>;
  createTag(name: string): Promise<Tag>;
  deleteAnnotation(example: AnnotationIdentity): Promise<void>;
  getExample(runId: string, sampleId: string): Promise<ExampleDetail>;
  getExamples(runId: string, query: ExampleQuery): Promise<ExamplesResponse>;
  getFailures(runId: string): Promise<FailuresResponse>;
  getReviewExamples(runId: string, query: ReviewExamplesQuery): Promise<ReviewExamplesResponse>;
  getRuns(): Promise<RunSummary[]>;
  getTags(): Promise<Tag[]>;
  getTaskAnnotation(identity: TaskAnnotationIdentity): Promise<TaskAnnotation | null>;
  getWaterfall(runId: string): Promise<WaterfallResponse>;
  putAnnotation(example: AnnotationIdentity, input: AnnotationInput): Promise<Annotation>;
  putTaskAnnotation(
    identity: TaskAnnotationIdentity,
    input: TaskAnnotationInput,
  ): Promise<TaskAnnotation>;
  deleteTaskAnnotation(identity: TaskAnnotationIdentity): Promise<void>;
}

interface FetchResponse {
  json(): Promise<unknown>;
  ok: boolean;
  status: number;
  statusText: string;
}

export type FetchTransport = (
  input: string,
  init?: { body?: string; headers?: Record<string, string>; method?: string },
) => Promise<FetchResponse>;

function defaultTransport(
  input: string,
  init?: Parameters<FetchTransport>[1],
): Promise<FetchResponse> {
  return globalThis.fetch(input, init);
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function queryString(values: object): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(values)) {
    if (value !== undefined) params.set(key, String(value));
  }
  const encoded = params.toString();
  return encoded === "" ? "" : `?${encoded}`;
}

function segment(value: string): string {
  return encodeURIComponent(value);
}

function detailMessage(payload: unknown): string | undefined {
  if (typeof payload !== "object" || payload === null || !("detail" in payload)) return undefined;
  const detail = payload.detail;
  return typeof detail === "string" ? detail : undefined;
}

export class HttpPreprocessingApi implements PreprocessingApi {
  constructor(
    private readonly transport: FetchTransport = defaultTransport,
    private readonly baseUrl = "",
  ) {}

  private async request<T>(path: string, init?: Parameters<FetchTransport>[1]): Promise<T> {
    const response = await this.transport(`${this.baseUrl}${path}`, init);
    const payload = response.status === 204 ? undefined : await response.json();
    if (!response.ok) {
      throw new ApiError(
        detailMessage(payload) ?? response.statusText ?? `Request failed (${response.status})`,
        response.status,
      );
    }
    return payload as T;
  }

  getRuns(): Promise<RunSummary[]> {
    return this.request("/api/runs");
  }

  getWaterfall(runId: string): Promise<WaterfallResponse> {
    return this.request(`/api/waterfall${queryString({ run_id: runId })}`);
  }

  getFailures(runId: string): Promise<FailuresResponse> {
    return this.request(`/api/failures${queryString({ run_id: runId })}`);
  }

  getExamples(runId: string, query: ExampleQuery): Promise<ExamplesResponse> {
    return this.request(`/api/examples${queryString({ run_id: runId, ...query })}`);
  }

  getExample(runId: string, sampleId: string): Promise<ExampleDetail> {
    return this.request(`/api/example${queryString({ run_id: runId, sample_id: sampleId })}`);
  }

  getReviewExamples(runId: string, query: ReviewExamplesQuery): Promise<ReviewExamplesResponse> {
    return this.request(`/api/review-examples${queryString({ run_id: runId, ...query })}`);
  }

  compare(baselineRunId: string, candidateRunId: string): Promise<CompareResponse> {
    return this.request(`/api/compare${queryString({ baseline: baselineRunId, candidate: candidateRunId })}`);
  }

  getTags(): Promise<Tag[]> {
    return this.request("/api/tags");
  }

  createTag(name: string): Promise<Tag> {
    return this.request("/api/tags", {
      body: JSON.stringify({ name }),
      headers: { "Content-Type": "application/json" },
      method: "POST",
    });
  }

  putAnnotation(
    example: AnnotationIdentity,
    input: AnnotationInput,
  ): Promise<Annotation> {
    return this.request(this.annotationPath(example), {
      body: JSON.stringify(input),
      headers: { "Content-Type": "application/json" },
      method: "PUT",
    });
  }

  deleteAnnotation(
    example: AnnotationIdentity,
  ): Promise<void> {
    return this.request(this.annotationPath(example), { method: "DELETE" });
  }

  async getTaskAnnotation(
    identity: TaskAnnotationIdentity,
  ): Promise<TaskAnnotation | null> {
    try {
      return await this.request(this.taskAnnotationPath(identity));
    } catch (error: unknown) {
      if (error instanceof ApiError && error.status === 404) return null;
      throw error;
    }
  }

  putTaskAnnotation(
    identity: TaskAnnotationIdentity,
    input: TaskAnnotationInput,
  ): Promise<TaskAnnotation> {
    return this.request(this.taskAnnotationPath(identity), {
      body: JSON.stringify(input),
      headers: { "Content-Type": "application/json" },
      method: "PUT",
    });
  }

  deleteTaskAnnotation(identity: TaskAnnotationIdentity): Promise<void> {
    return this.request(this.taskAnnotationPath(identity), {
      method: "DELETE",
    });
  }

  private annotationPath(
    example: AnnotationIdentity,
  ): string {
    return `/api/annotations/${segment(example.corpus_sha256)}/${segment(example.decoder_output_sha256)}${queryString({ sample_id: example.sample_id })}`;
  }

  private taskAnnotationPath(identity: TaskAnnotationIdentity): string {
    return `/api/task-annotations${queryString(identity)}`;
  }
}

export const defaultApi = new HttpPreprocessingApi();

export function annotationExportUrl(baseUrl = ""): string {
  return `${baseUrl}/api/annotations/export`;
}

export function taskAnnotationExportUrl(baseUrl = ""): string {
  return `${baseUrl}/api/task-annotations/export`;
}
