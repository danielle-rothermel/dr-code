import data from "./data/viewer-data.json";

export type Denominator = "all" | "present" | "nonblank";

export interface FunnelStage {
  count: number;
  rate: number;
  rate_label: string;
  stage: string;
  unit: "sample" | "candidate_row";
}

export interface Outcome {
  count_all: number;
  count_nonblank: number;
  count_present: number;
  outcome: string;
  rate_of_all: number;
  rate_of_nonblank: number;
  rate_of_present: number;
}

export interface Candidate {
  candidate_id: string;
  candidate_index: number;
  cleaned_source: string;
  compile_warnings: string[];
  origins: Array<{ strategy: string; variant: string }>;
  top_level_function_names: string[];
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

export interface Example {
  candidates: Candidate[];
  categories: string[];
  context: Record<string, string | null>;
  facts: DiagnosticFact[];
  final_candidate_count: number;
  outcome: string;
  raw_decoder_output: string | null;
  rejections: Rejection[];
  sample_id: string;
}

export interface FailureBrowserGroup {
  count: number;
  failed_step: string;
  failure_code: string;
  index_path: string;
}

export interface FailureBrowserSummary {
  artifact_id: string;
  groups: FailureBrowserGroup[];
  schema_version: 1;
  total_count: number;
}

export interface FailureIndexEntry {
  context: Record<string, string | null>;
  detail_shard: string;
  failed_step: string;
  outcome: string;
  raw_character_count: number;
  rejection_reasons: string[];
  sample_id: string;
}

export interface FailureGroupIndex {
  count: number;
  entries: FailureIndexEntry[];
  failed_step: string;
  failure_code: string;
  schema_version: 1;
}

export interface FailureDetailShard {
  examples: Example[];
  failure_code: string;
  schema_version: 1;
}

export interface OutcomeByDimension {
  count: number;
  denominator: Denominator;
  denominator_count: number;
  dimension: string;
  outcome: string;
  rate: number;
  value: string;
}

export type TestOutcome =
  | "passed"
  | "failed"
  | "timed_out"
  | "infrastructure_failure";

export interface EvaluationFunnelStage {
  count: number;
  denominator: "extracted_final_candidates";
  denominator_count: number;
  rate: number | null;
  stage: string;
  unit: "candidate_row";
}

export interface CandidateTestOutcome {
  candidate_count: number;
  candidate_denominator_count: number;
  candidate_rate: number | null;
  failure_type: string;
  official_outcome: string;
  record_status: string;
  test_outcome: TestOutcome;
}

export interface SampleBestTestOutcome {
  best_test_outcome: TestOutcome;
  extracted_sample_count: number;
  rate_of_extracted_samples: number | null;
  sample_count: number;
}

export interface EvaluationComparison {
  dimension: string;
  error_count?: number;
  evaluated_sample_count: number;
  failed_count: number;
  infrastructure_failure_count: number;
  not_extracted_count?: number;
  pass_rate_of_all_samples: number | null;
  pass_rate_of_evaluated_samples: number | null;
  passed_count: number;
  sample_count: number;
  timed_out_count: number;
  value: string;
}

export interface OriginTestSuccess {
  candidate_origin_count: number;
  failed_count: number;
  infrastructure_failure_count: number;
  pass_rate: number | null;
  passed_count: number;
  strategy: string;
  timed_out_count: number;
  unit: "final_candidate_origin_attribution";
  variant: string;
}

export interface EvaluationExample {
  candidate_id: string;
  candidate_index: number;
  categories: string[];
  cleaned_source: string;
  context: Record<string, string | null>;
  diagnostics: {
    best_function_name: string | null;
    coverage_complete: boolean | null;
    error_count: number | null;
    failed_count: number | null;
    failure_message: string | null;
    failure_type: string | null;
    function_count: number | null;
    passed_count: number | null;
    timeout_count: number | null;
    total_cases: number | null;
  };
  evaluation_key: string;
  official_outcome: string | null;
  origins: Array<{ strategy: string; variant: string }>;
  record_status: string;
  sample_id: string;
  task_id: string;
  test_outcome: TestOutcome;
}

export interface CandidateEvaluation {
  examples: EvaluationExample[];
  summary: {
    available: true;
    candidate_membership_count: number;
    candidate_outcomes: CandidateTestOutcome[];
    deduplicated_evaluation_count: number;
    extracted_sample_count: number;
    funnel: EvaluationFunnelStage[];
    limitations?: string[];
    provenance?: {
      manifest?: {
        complete: boolean;
        label: string;
        membership_rows: number;
        result_rows: number;
        schema_version: number;
        sha256: string;
      };
      semantic_coordinates: {
        metrics_profile?: string;
        operator?: string;
        runner_identity?: string;
        sandbox_image?: string | null;
        snapshot_sha256?: string;
      };
    };
    sample_best_outcomes: SampleBestTestOutcome[];
  };
  test_success_by_dimension: EvaluationComparison[];
  test_success_by_multiplicity: EvaluationComparison[];
  test_success_by_origin: OriginTestSuccess[];
  test_success_by_preprocessing_outcome: EvaluationComparison[];
}

export interface ViewerData {
  candidate_evaluation?: CandidateEvaluation;
  candidate_multiplicity: Array<{
    denominator: Denominator;
    denominator_count: number;
    final_candidate_count: number;
    rate: number;
    sample_count: number;
  }>;
  compile_warnings: Array<{
    candidate_count: number;
    candidate_rate: number;
    warning: string;
  }>;
  examples: Example[];
  failure_browser?: FailureBrowserSummary;
  failure_modes: Array<{
    count: number;
    failed_step: string;
    reason: string;
    sample_count: number;
    scope: string;
  }>;
  headline: {
    candidate_invariants: Record<string, number>;
    denominators: Record<Denominator, number>;
    funnel: FunnelStage[];
    outcomes: Outcome[];
  };
  origin_contribution: Array<{
    converged_final_candidate_count: number;
    convergence_rate: number | null;
    extracted_candidate_count: number;
    final_candidate_origin_count: number;
    recovery_rate: number | null;
    strategy: string;
    variant: string;
  }>;
  outcome_by_dimension: OutcomeByDimension[];
  schema_version: number;
}

interface JsonResponse {
  json(): Promise<unknown>;
  ok: boolean;
  status: number;
}

export type FailureArtifactFetch = (path: string) => Promise<JsonResponse>;

type JsonRecord = Record<string, unknown>;

function requireRecord(value: unknown, label: string): JsonRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
  return value as JsonRecord;
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== "string") throw new Error(`${label} must be a string`);
  return value;
}

function requireNullableString(value: unknown, label: string): string | null {
  if (value === null) return null;
  return requireString(value, label);
}

function requireNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number`);
  }
  return value;
}

function requireArray(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array`);
  return value;
}

function parseStringArray(value: unknown, label: string): string[] {
  return requireArray(value, label).map((item, index) =>
    requireString(item, `${label}[${index}]`),
  );
}

function parseContext(value: unknown, label: string): Record<string, string | null> {
  const record = requireRecord(value, label);
  return Object.fromEntries(
    Object.entries(record).map(([key, item]) => [
      key,
      requireNullableString(item, `${label}.${key}`),
    ]),
  );
}

function parseCandidate(value: unknown, label: string): Candidate {
  const candidate = requireRecord(value, label);
  return {
    candidate_id: requireString(candidate.candidate_id, `${label}.candidate_id`),
    candidate_index: requireNumber(candidate.candidate_index, `${label}.candidate_index`),
    cleaned_source: requireString(candidate.cleaned_source, `${label}.cleaned_source`),
    compile_warnings: parseStringArray(candidate.compile_warnings, `${label}.compile_warnings`),
    origins: requireArray(candidate.origins, `${label}.origins`).map((value, index) => {
      const origin = requireRecord(value, `${label}.origins[${index}]`);
      return {
        strategy: requireString(origin.strategy, `${label}.origins[${index}].strategy`),
        variant: requireString(origin.variant, `${label}.origins[${index}].variant`),
      };
    }),
    top_level_function_names: parseStringArray(
      candidate.top_level_function_names,
      `${label}.top_level_function_names`,
    ),
  };
}

function parseExample(value: unknown, label: string): Example {
  const example = requireRecord(value, label);
  return {
    candidates: requireArray(example.candidates, `${label}.candidates`).map((item, index) =>
      parseCandidate(item, `${label}.candidates[${index}]`),
    ),
    categories: parseStringArray(example.categories, `${label}.categories`),
    context: parseContext(example.context, `${label}.context`),
    facts: requireArray(example.facts, `${label}.facts`).map((item, index) => {
      const fact = requireRecord(item, `${label}.facts[${index}]`);
      return {
        facts_json: requireString(fact.facts_json, `${label}.facts[${index}].facts_json`),
        step_name: requireString(fact.step_name, `${label}.facts[${index}].step_name`),
      };
    }),
    final_candidate_count: requireNumber(
      example.final_candidate_count,
      `${label}.final_candidate_count`,
    ),
    outcome: requireString(example.outcome, `${label}.outcome`),
    raw_decoder_output: requireNullableString(
      example.raw_decoder_output,
      `${label}.raw_decoder_output`,
    ),
    rejections: requireArray(example.rejections, `${label}.rejections`).map((item, index) => {
      const rejection = requireRecord(item, `${label}.rejections[${index}]`);
      return {
        details_json: requireString(
          rejection.details_json,
          `${label}.rejections[${index}].details_json`,
        ),
        reason_code: requireNullableString(
          rejection.reason_code,
          `${label}.rejections[${index}].reason_code`,
        ),
        step_name: requireString(
          rejection.step_name,
          `${label}.rejections[${index}].step_name`,
        ),
      };
    }),
    sample_id: requireString(example.sample_id, `${label}.sample_id`),
  };
}

function parseFailureIndex(value: unknown): FailureGroupIndex {
  const index = requireRecord(value, "Failure group index");
  if (index.schema_version !== 1) {
    throw new Error("Failure group index has an unsupported schema version");
  }
  const entries = requireArray(index.entries, "Failure group index.entries").map((item, position) => {
    const entry = requireRecord(item, `Failure group index.entries[${position}]`);
    return {
      context: parseContext(entry.context, `Failure group index.entries[${position}].context`),
      detail_shard: requireString(
        entry.detail_shard,
        `Failure group index.entries[${position}].detail_shard`,
      ),
      failed_step: requireString(
        entry.failed_step,
        `Failure group index.entries[${position}].failed_step`,
      ),
      outcome: requireString(entry.outcome, `Failure group index.entries[${position}].outcome`),
      raw_character_count: requireNumber(
        entry.raw_character_count,
        `Failure group index.entries[${position}].raw_character_count`,
      ),
      rejection_reasons: parseStringArray(
        entry.rejection_reasons,
        `Failure group index.entries[${position}].rejection_reasons`,
      ),
      sample_id: requireString(
        entry.sample_id,
        `Failure group index.entries[${position}].sample_id`,
      ),
    };
  });
  const count = requireNumber(index.count, "Failure group index.count");
  if (count !== entries.length) {
    throw new Error(`Failure group index expected ${count} entries but contained ${entries.length}`);
  }
  return {
    count,
    entries,
    failed_step: requireString(index.failed_step, "Failure group index.failed_step"),
    failure_code: requireString(index.failure_code, "Failure group index.failure_code"),
    schema_version: 1,
  };
}

function parseFailureDetailShard(value: unknown): FailureDetailShard {
  const shard = requireRecord(value, "Failure detail shard");
  if (shard.schema_version !== 1) {
    throw new Error("Failure detail shard has an unsupported schema version");
  }
  return {
    examples: requireArray(shard.examples, "Failure detail shard.examples").map((item, index) =>
      parseExample(item, `Failure detail shard.examples[${index}]`),
    ),
    failure_code: requireString(shard.failure_code, "Failure detail shard.failure_code"),
    schema_version: 1,
  };
}

function failureArtifactPath(relativePath: string, baseUrl: string): string {
  const segments = relativePath.split("/");
  if (
    relativePath === "" ||
    relativePath.startsWith("/") ||
    segments.some((segment) => segment === "" || segment === "." || segment === "..")
  ) {
    throw new Error(`Invalid failure artifact path: ${relativePath}`);
  }
  const normalizedBase = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  return `${normalizedBase}data/failure-examples/${segments.map(encodeURIComponent).join("/")}`;
}

export class FailureExamplesLoader {
  private readonly cache = new Map<string, Promise<unknown>>();

  constructor(
    private readonly fetchArtifact: FailureArtifactFetch = (path) => fetch(path),
    private readonly baseUrl: string = import.meta.env.BASE_URL,
  ) {}

  async loadIndex(group: FailureBrowserGroup): Promise<FailureGroupIndex> {
    const path = failureArtifactPath(group.index_path, this.baseUrl);
    const index = await this.load(path, parseFailureIndex);
    if (
      index.failure_code !== group.failure_code ||
      index.failed_step !== group.failed_step ||
      index.count !== group.count
    ) {
      throw new Error(`Failure group index metadata does not match ${group.failure_code}`);
    }
    return index;
  }

  async loadDetail(
    entry: FailureIndexEntry,
    failureCode: string,
  ): Promise<Example> {
    const path = failureArtifactPath(entry.detail_shard, this.baseUrl);
    const shard = await this.load(path, parseFailureDetailShard);
    if (shard.failure_code !== failureCode) {
      throw new Error(`Failure detail shard metadata does not match ${failureCode}`);
    }
    const example = shard.examples.find(({ sample_id }) => sample_id === entry.sample_id);
    if (example === undefined) {
      throw new Error(`Failure detail shard does not contain sample ${entry.sample_id}`);
    }
    return example;
  }

  private load<T>(path: string, parse: (value: unknown) => T): Promise<T> {
    const cached = this.cache.get(path) as Promise<T> | undefined;
    if (cached !== undefined) return cached;

    const request = this.fetchArtifact(path)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Could not load ${path} (HTTP ${response.status})`);
        }
        return parse(await response.json());
      })
      .catch((error: unknown) => {
        this.cache.delete(path);
        throw error;
      });
    this.cache.set(path, request);
    return request;
  }
}

export const FAILURE_PAGE_SIZE = 50;

export function filterFailureEntries(
  entries: FailureIndexEntry[],
  search: string,
): FailureIndexEntry[] {
  const normalizedSearch = search.trim().toLocaleLowerCase();
  if (normalizedSearch === "") return entries;

  return entries.filter((entry) => {
    const searchableValues = [
      entry.sample_id,
      entry.outcome,
      entry.failed_step,
      ...entry.rejection_reasons,
      ...Object.entries(entry.context).flatMap(([key, value]) => [key, value ?? ""]),
    ];
    return [...searchableValues, ...searchableValues.map((value) => value.replaceAll("_", " "))]
      .join(" ")
      .toLocaleLowerCase()
      .includes(normalizedSearch);
  });
}

export interface FailurePage {
  end: number;
  entries: FailureIndexEntry[];
  page: number;
  pageCount: number;
  start: number;
  total: number;
}

export function paginateFailureEntries(
  entries: FailureIndexEntry[],
  requestedPage: number,
): FailurePage {
  const pageCount = Math.max(1, Math.ceil(entries.length / FAILURE_PAGE_SIZE));
  const page = Math.min(Math.max(1, requestedPage), pageCount);
  const offset = (page - 1) * FAILURE_PAGE_SIZE;
  const pageEntries = entries.slice(offset, offset + FAILURE_PAGE_SIZE);
  return {
    end: offset + pageEntries.length,
    entries: pageEntries,
    page,
    pageCount,
    start: pageEntries.length === 0 ? 0 : offset + 1,
    total: entries.length,
  };
}

// The generation artifact is checked in and refreshed with `pnpm data:sync`.
// This explicit cast keeps the artifact boundary local to the app.
export const viewerData = data as ViewerData;

export const dimensions = Array.from(
  new Set(viewerData.outcome_by_dimension.map((row) => row.dimension)),
).sort();

export const outcomeNames = Array.from(
  new Set(viewerData.outcome_by_dimension.map((row) => row.outcome)),
).sort();

export function filterExamples(
  examples: Example[],
  search: string,
  outcome: string,
): Example[] {
  const normalizedSearch = search.trim().toLocaleLowerCase();

  return examples.filter((example) => {
    if (outcome !== "all" && example.outcome !== outcome) return false;
    if (normalizedSearch === "") return true;

    const searchable = [
      example.sample_id,
      example.outcome,
      ...example.categories,
      ...Object.values(example.context).map((value) => value ?? ""),
    ]
      .join(" ")
      .toLocaleLowerCase();
    return searchable.includes(normalizedSearch);
  });
}

export function crosstabRows(
  dimension: string,
  outcome: string,
  denominator: Denominator,
): OutcomeByDimension[] {
  return viewerData.outcome_by_dimension
    .filter(
      (row) =>
        row.dimension === dimension &&
        row.outcome === outcome &&
        row.denominator === denominator,
    )
    .sort((left, right) => right.count - left.count || left.value.localeCompare(right.value));
}

export function filterEvaluationExamples(
  examples: EvaluationExample[],
  search: string,
  testOutcome: TestOutcome | "all",
): EvaluationExample[] {
  const normalizedSearch = search.trim().toLocaleLowerCase();

  return examples.filter((example) => {
    if (testOutcome !== "all" && example.test_outcome !== testOutcome) {
      return false;
    }
    if (normalizedSearch === "") return true;

    const searchable = [
      example.sample_id,
      example.candidate_id,
      example.evaluation_key,
      example.task_id,
      example.test_outcome,
      example.official_outcome ?? "",
      example.record_status,
      example.diagnostics.best_function_name ?? "",
      example.diagnostics.failure_type ?? "",
      example.diagnostics.failure_message ?? "",
      ...example.categories,
      ...example.origins.flatMap(({ strategy, variant }) => [strategy, variant]),
      ...Object.values(example.context).map((value) => value ?? ""),
    ]
      .join(" ")
      .toLocaleLowerCase();
    return searchable.includes(normalizedSearch);
  });
}
