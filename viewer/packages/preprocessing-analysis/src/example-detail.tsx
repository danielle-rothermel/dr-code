import { CodeBlock, CodeDiff, StatusBadge, type StatusBadgeStatus } from "@dr-code/viewer";

import type { ExampleDetail as ExampleDetailModel } from "./api";
import { humanize } from "./format";

function outcomeStatus(outcome: string): StatusBadgeStatus {
  if (outcome.includes("passing") || outcome.includes("function_candidate")) return "success";
  if (outcome.includes("missing") || outcome.includes("blank")) return "warning";
  return "failure";
}

function DiagnosticList({ example }: { example: ExampleDetailModel }) {
  if (example.facts.length === 0 && example.rejections.length === 0) {
    return <p className="empty-state">No preprocessing diagnostics were recorded.</p>;
  }

  return (
    <div className="diagnostics">
      {example.rejections.map((rejection, index) => (
        <article className="diagnostic diagnostic--rejection" key={`${rejection.step_name}-${index}`}>
          <div className="diagnostic__heading">
            <StatusBadge status="failure">rejected</StatusBadge>
            <strong>{humanize(rejection.step_name)}</strong>
            <span>{rejection.reason_code === null ? "reason unavailable" : humanize(rejection.reason_code)}</span>
          </div>
          <CodeBlock code={rejection.details_json} lang="json" />
        </article>
      ))}
      {example.facts.map((fact, index) => (
        <article className="diagnostic" key={`${fact.step_name}-${index}`}>
          <div className="diagnostic__heading">
            <StatusBadge status="neutral">step</StatusBadge>
            <strong>{humanize(fact.step_name)}</strong>
          </div>
          <CodeBlock code={fact.facts_json} lang="json" />
        </article>
      ))}
    </div>
  );
}

export function ExampleDetail({
  example,
  eyebrow = "Example detail",
  titleId = "example-detail-title",
}: {
  example: ExampleDetailModel;
  eyebrow?: string;
  titleId?: string;
}) {
  const firstCandidate = example.candidates[0];
  const showDiff =
    firstCandidate !== undefined &&
    example.raw_decoder_output !== null &&
    example.raw_decoder_output.length < 8_000;

  return (
    <article className="example-detail" aria-labelledby={titleId}>
      <div className="detail-heading">
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h3 id={titleId}>{example.sample_id}</h3>
        </div>
        <StatusBadge status={outcomeStatus(example.outcome)}>{humanize(example.outcome)}</StatusBadge>
      </div>

      {Object.keys(example.context).length > 0 && (
        <dl className="metadata-grid">
          {Object.entries(example.context).map(([key, value]) => (
            <div key={key}>
              <dt>{humanize(key)}</dt>
              <dd>{value === null ? "not recorded" : String(value)}</dd>
            </div>
          ))}
        </dl>
      )}

      <div className="code-columns">
        <section aria-labelledby="decoder-output-title">
          <h4 id="decoder-output-title">Decoder output</h4>
          {example.raw_decoder_output === null ? (
            <p className="empty-state">No decoder output was present.</p>
          ) : (
            <CodeBlock code={example.raw_decoder_output} lang="python" />
          )}
        </section>
        <section aria-labelledby="candidate-output-title">
          <h4 id="candidate-output-title">Final candidates</h4>
          {example.candidates.length === 0 ? (
            <p className="empty-state">No candidate survived preprocessing.</p>
          ) : example.candidates.map((candidate) => (
            <article className="candidate" key={candidate.candidate_id}>
              <div className="candidate__heading">
                <strong>Candidate {candidate.candidate_index + 1}</strong>
                <span>{candidate.top_level_function_names.join(", ") || "no named function"}</span>
              </div>
              <CodeBlock code={candidate.cleaned_source} lang="python" />
              {candidate.origins.length > 0 && (
                <small>
                  {candidate.origins.map(({ strategy, variant }) => `${humanize(strategy)} / ${humanize(variant)}`).join(" · ")}
                </small>
              )}
              {candidate.compile_warnings.length > 0 && (
                <p className="warning-copy">Warnings: {candidate.compile_warnings.join("; ")}</p>
              )}
            </article>
          ))}
        </section>
      </div>

      {showDiff && firstCandidate !== undefined && (
        <details className="detail-section">
          <summary>Compare decoder output with first candidate</summary>
          <CodeDiff
            lang="python"
            mode="unified"
            newContent={firstCandidate.cleaned_source}
            newName="first candidate"
            oldContent={example.raw_decoder_output ?? ""}
            oldName="decoder output"
          />
        </details>
      )}

      <details className="detail-section" open={example.rejections.length > 0}>
        <summary>
          Diagnostics ({example.facts.length} facts, {example.rejections.length} rejections)
        </summary>
        <DiagnosticList example={example} />
      </details>
    </article>
  );
}
