/**
 * @dr-code/viewer — the canonical React components for code
 * visualization (ADR 0006). Consumers import these components only;
 * shiki and @git-diff-view are wrapped implementation details.
 */
export { CodeBlock, type CodeBlockProps } from "./code-block.js";
export {
  CodeBlockClient,
  type CodeBlockClientProps,
} from "./code-block-client.js";
export {
  TransformDiff,
  type TransformDiffMode,
  type TransformDiffProps,
  type TransformDiffTheme,
} from "./transform-diff.js";
export {
  ExtractionTraceView,
  type ExtractionTraceViewProps,
} from "./extraction-trace-view.js";
export {
  EvaluationCaseTable,
  type EvaluationCaseTableProps,
} from "./evaluation-case-table.js";
export { TaskCard, type TaskCardProps } from "./task-card.js";
export type {
  CandidateSelectionTrace,
  CandidateStatus,
  CodeParserProfile,
  EvaluationCaseStatus,
  EvaluationCaseSummary,
  ExtractionMethod,
  ExtractionTrace,
  ExtractionTraceNode,
  HumanEvalTask,
  HumanEvalTestCaseKind,
  TraceCheckVerdict,
  TraceNodeKind,
} from "./types.js";
