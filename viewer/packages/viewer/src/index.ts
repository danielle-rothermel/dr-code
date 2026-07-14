/**
 * @dr-code/viewer — the canonical React components for code
 * visualization. Consumers import these components only; shiki and
 * @git-diff-view are wrapped implementation details.
 */
export { CodeBlock, type CodeBlockProps } from "./code-block.js";
export {
  CodeDiff,
  type CodeDiffMode,
  type CodeDiffProps,
  type CodeDiffTheme,
} from "./code-diff.js";
export {
  ExtractionTraceView,
  type ExtractionTraceViewProps,
} from "./extraction-trace-view.js";
export {
  EvaluationCaseTable,
  type EvaluationCaseTableProps,
} from "./evaluation-case-table.js";
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
