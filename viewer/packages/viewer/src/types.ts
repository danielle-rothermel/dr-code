/**
 * Public domain types.
 *
 * Extraction-trace shapes are generated from the serve OpenAPI schema
 * (`pnpm gen:serve`); library shapes from the pydantic JSON-schema dump
 * (`pnpm gen:humaneval`). Regenerate instead of editing `src/gen/`.
 */
import type { components } from "./gen/serve.js";

export type ExtractionTrace = components["schemas"]["ExtractionTrace"];
export type ExtractionTraceNode = components["schemas"]["ExtractionTraceNode"];
export type CandidateSelectionTrace =
  components["schemas"]["CandidateSelectionTrace"];
export type CodeParserProfile = components["schemas"]["CodeParserProfile"];
export type TraceNodeKind = components["schemas"]["TraceNodeKind"];
export type TraceCheckVerdict = components["schemas"]["TraceCheckVerdict"];
export type CandidateStatus = components["schemas"]["CandidateStatus"];
export type ExtractionMethod = components["schemas"]["ExtractionMethod"];

export type {
  EvaluationCaseStatus,
  EvaluationCaseSummary,
  HumanEvalTask,
  HumanEvalTestCaseKind,
} from "./gen/humaneval.js";
