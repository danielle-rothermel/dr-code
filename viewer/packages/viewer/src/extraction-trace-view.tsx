"use client";

import { CodeBlockClient } from "./code-block-client.js";
import { DEFAULT_LANGUAGE } from "./themes.js";
import { TransformDiff } from "./transform-diff.js";
import type {
  CandidateSelectionTrace,
  ExtractionTrace,
  ExtractionTraceNode,
} from "./types.js";

export interface ExtractionTraceViewProps {
  trace: ExtractionTrace;
  lang?: string;
}

/**
 * Renders the parser's lineage-tree extraction trace: fork nodes,
 * transform nodes with before/after diffs, check verdicts, and the
 * candidate selection walk.
 */
export function ExtractionTraceView({
  trace,
  lang = DEFAULT_LANGUAGE,
}: ExtractionTraceViewProps) {
  return (
    <section className="drv-extraction-trace">
      <header className="drv-trace-header">
        <span className="drv-trace-profile">
          {trace.profile.profile_id}@{trace.profile.version}
        </span>
        {trace.extraction_method != null && (
          <span className="drv-trace-method">{trace.extraction_method}</span>
        )}
        <p className="drv-trace-rationale">{trace.rationale}</p>
        {trace.extraction_error != null && (
          <p className="drv-trace-error">{trace.extraction_error}</p>
        )}
      </header>
      <div className="drv-trace-tree">
        {trace.roots.map((node, index) => (
          <TraceNode key={index} node={node} lang={lang} />
        ))}
      </div>
      {trace.candidates.length > 0 && (
        <ol className="drv-trace-candidates">
          {trace.candidates.map((candidate) => (
            <li key={candidate.index}>
              <CandidateEntry
                candidate={candidate}
                selected={candidate.index === trace.selected_candidate_index}
                lang={lang}
              />
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

function TraceNode({
  node,
  lang,
}: {
  node: ExtractionTraceNode;
  lang: string;
}) {
  return (
    <details className={`drv-trace-node drv-trace-node-${node.kind}`} open>
      <summary>
        <span className="drv-trace-kind">{node.kind}</span>
        <span className="drv-trace-name">{node.name}</span>
        {node.verdict != null && (
          <span className={`drv-trace-verdict drv-trace-verdict-${node.verdict}`}>
            {node.verdict}
          </span>
        )}
        {node.reason != null && (
          <span className="drv-trace-reason">{node.reason}</span>
        )}
      </summary>
      <NodeBody node={node} lang={lang} />
      {node.children != null && node.children.length > 0 && (
        <div className="drv-trace-children">
          {node.children.map((child, index) => (
            <TraceNode key={index} node={child} lang={lang} />
          ))}
        </div>
      )}
    </details>
  );
}

function NodeBody({ node, lang }: { node: ExtractionTraceNode; lang: string }) {
  if (node.kind === "transform" && node.before_text != null && node.after_text != null) {
    if (node.before_text === node.after_text) {
      return <p className="drv-trace-unchanged">unchanged</p>;
    }
    return (
      <TransformDiff
        oldContent={node.before_text}
        newContent={node.after_text}
        oldName={`${node.name}.before`}
        newName={`${node.name}.after`}
        lang={lang}
      />
    );
  }
  const text = node.after_text ?? node.before_text;
  if (text == null || text === "") return null;
  return <CodeBlockClient code={text} lang={lang} />;
}

function CandidateEntry({
  candidate,
  selected,
  lang,
}: {
  candidate: CandidateSelectionTrace;
  selected: boolean;
  lang: string;
}) {
  const classes = selected
    ? "drv-trace-candidate drv-trace-candidate-selected"
    : "drv-trace-candidate";
  return (
    <div className={classes}>
      <header>
        <span className="drv-trace-candidate-index">
          candidate {candidate.index}
        </span>
        <span
          className={`drv-trace-candidate-status drv-trace-candidate-status-${candidate.status}`}
        >
          {candidate.status}
        </span>
        {candidate.rejection_reason != null && (
          <span className="drv-trace-reason">{candidate.rejection_reason}</span>
        )}
      </header>
      {candidate.checks != null && candidate.checks.length > 0 && (
        <div className="drv-trace-checks">
          {candidate.checks.map((check, index) => (
            <TraceNode key={index} node={check} lang={lang} />
          ))}
        </div>
      )}
      <CodeBlockClient code={candidate.source} lang={lang} />
    </div>
  );
}
