import { useMemo, useState } from "react";

import { CodeBlock, CodeDiff, StatusBadge, type StatusBadgeStatus } from "@dr-code/viewer";

import {
  crosstabRows,
  dimensions,
  FailureExamplesLoader,
  filterExamples,
  outcomeNames,
  type Denominator,
  type Example,
  type ViewerData,
  viewerData,
} from "./data";
import { EvaluationAnalysis, EvaluationUnavailable } from "./evaluation";
import { FailureBrowser } from "./failure-browser";

const defaultFailureLoader = new FailureExamplesLoader();

const numberFormatter = new Intl.NumberFormat("en-US");
const percentFormatter = new Intl.NumberFormat("en-US", {
  style: "percent",
  maximumFractionDigits: 2,
});

function formatNumber(value: number): string {
  return numberFormatter.format(value);
}

function formatPercent(value: number | null): string {
  return value === null ? "—" : percentFormatter.format(value);
}

function humanize(value: string): string {
  return value.replaceAll("_", " ");
}

function outcomeStatus(outcome: string): StatusBadgeStatus {
  if (outcome === "function_candidates_extracted") return "success";
  if (outcome.includes("missing") || outcome.includes("blank")) return "warning";
  return "failure";
}

function CandidateCount({ count }: { count: number }) {
  return (
    <StatusBadge status={count > 0 ? "success" : "failure"}>
      {count} final {count === 1 ? "candidate" : "candidates"}
    </StatusBadge>
  );
}

function OutcomeBadge({ outcome }: { outcome: string }) {
  return <StatusBadge status={outcomeStatus(outcome)}>{humanize(outcome)}</StatusBadge>;
}

function FactList({ example }: { example: Example }) {
  if (example.facts.length === 0 && example.rejections.length === 0) {
    return <p className="empty-copy">No preprocessing diagnostics were recorded for this spot check.</p>;
  }

  return (
    <div className="diagnostics">
      {example.rejections.map((rejection, index) => (
        <article className="diagnostic diagnostic--rejection" key={`${rejection.step_name}-${index}`}>
          <div>
            <StatusBadge status="failure">rejected</StatusBadge>
            <strong>{humanize(rejection.step_name)}</strong>
            <span>{rejection.reason_code === null ? "reason unavailable" : humanize(rejection.reason_code)}</span>
          </div>
          <CodeBlock code={rejection.details_json} lang="json" className="diagnostic__code" />
        </article>
      ))}
      {example.facts.map((fact) => (
        <article className="diagnostic" key={fact.step_name}>
          <div>
            <StatusBadge status="neutral">step</StatusBadge>
            <strong>{humanize(fact.step_name)}</strong>
          </div>
          <CodeBlock code={fact.facts_json} lang="json" className="diagnostic__code" />
        </article>
      ))}
    </div>
  );
}

export function SpotCheck({
  example,
  eyebrow = "Active spot check",
  titleId = "spot-check-title",
}: {
  example: Example;
  eyebrow?: string;
  titleId?: string;
}) {
  const firstCandidate = example.candidates[0];
  const canDiff =
    firstCandidate !== undefined &&
    example.raw_decoder_output !== null &&
    example.raw_decoder_output.trim() !== "" &&
    example.raw_decoder_output.length < 4000;

  return (
    <article className="spot-check-detail" aria-labelledby={titleId}>
      <div className="spot-check-detail__heading">
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h3 id={titleId}>{example.sample_id}</h3>
        </div>
        <div className="badge-row">
          <OutcomeBadge outcome={example.outcome} />
          <CandidateCount count={example.final_candidate_count} />
        </div>
      </div>

      <dl className="metadata-grid">
        {Object.entries(example.context).map(([key, value]) => (
          <div key={key}>
            <dt>{humanize(key)}</dt>
            <dd>{value ?? "not recorded"}</dd>
          </div>
        ))}
      </dl>

      <div className="category-row" aria-label="Example categories">
        {example.categories.length > 0 ? example.categories.map((category) => (
          <span key={category}>{humanize(category)}</span>
        )) : <span>no sampled category tags</span>}
      </div>

      <div className="spot-check-columns">
        <section aria-labelledby="response-title">
          <h4 id="response-title">Decoder response</h4>
          {example.raw_decoder_output ? (
            <CodeBlock code={example.raw_decoder_output} lang="python" className="analysis-code" />
          ) : (
            <p className="empty-copy">No decoder response was present in this sample.</p>
          )}
        </section>
        <section aria-labelledby="candidates-title">
          <h4 id="candidates-title">Final candidates</h4>
          {example.candidates.length > 0 ? example.candidates.map((candidate) => (
            <article className="candidate" key={candidate.candidate_id}>
              <div className="candidate__meta">
                <span>candidate {candidate.candidate_index + 1}</span>
                <span>{candidate.top_level_function_names.join(", ") || "no named top-level function"}</span>
              </div>
              <CodeBlock code={candidate.cleaned_source} lang="python" className="analysis-code" />
              <p className="origin-line">
                Origin: {candidate.origins.map(({ strategy, variant }) => `${strategy} / ${variant}`).join(" · ")}
              </p>
              {candidate.compile_warnings.length > 0 && (
                <p className="warning-line">Warnings: {candidate.compile_warnings.join("; ")}</p>
              )}
            </article>
          )) : (
            <p className="empty-copy">No candidate survived preprocessing.</p>
          )}
        </section>
      </div>

      {canDiff && firstCandidate !== undefined && (
        <details className="source-diff">
          <summary>Inspect response → first final candidate</summary>
          <p>
            This comparison spans extraction and cleanup; it is not a claim that every changed line was a direct rewrite.
          </p>
          <CodeDiff
            oldContent={example.raw_decoder_output ?? ""}
            newContent={firstCandidate.cleaned_source}
            oldName="decoder response"
            newName="final candidate"
            lang="python"
            mode="unified"
          />
        </details>
      )}

      <details className="diagnostic-details" open={example.rejections.length > 0}>
        <summary>Preprocessing diagnostics ({example.facts.length} steps, {example.rejections.length} rejections)</summary>
        <FactList example={example} />
      </details>
    </article>
  );
}

export function Analysis({
  data = viewerData,
  failureLoader = defaultFailureLoader,
}: {
  data?: ViewerData;
  failureLoader?: FailureExamplesLoader;
}) {
  const [multiplicityDenominator, setMultiplicityDenominator] = useState<Denominator>("nonblank");
  const [crosstabDimension, setCrosstabDimension] = useState("source_kind");
  const [crosstabOutcome, setCrosstabOutcome] = useState("function_candidates_extracted");
  const [crosstabDenominator, setCrosstabDenominator] = useState<Denominator>("all");
  const [search, setSearch] = useState("");
  const [exampleOutcome, setExampleOutcome] = useState("all");
  const [selectedId, setSelectedId] = useState(data.examples[0]?.sample_id ?? "");

  const examples = useMemo(
    () => filterExamples(data.examples, search, exampleOutcome),
    [data.examples, exampleOutcome, search],
  );
  const selectedExample = examples.find((example) => example.sample_id === selectedId) ?? examples[0];
  const crosstab = useMemo(
    () => crosstabRows(crosstabDimension, crosstabOutcome, crosstabDenominator),
    [crosstabDenominator, crosstabDimension, crosstabOutcome],
  );
  const multiplicity = data.candidate_multiplicity.filter(
    (row) => row.denominator === multiplicityDenominator,
  );
  const extracted = data.headline.outcomes.find(
    ({ outcome }) => outcome === "function_candidates_extracted",
  );
  const noCandidates = data.candidate_multiplicity.find(
    (row) => row.denominator === "all" && row.final_candidate_count === 0,
  );
  const missing = data.headline.outcomes.find(
    ({ outcome }) => outcome === "decoder_output_missing",
  );

  return (
    <main>
      <header className="hero">
        <div>
          <p className="eyebrow">generation corpus · preprocessing audit</p>
          <h1>What survived the path from decoder output to usable functions?</h1>
          <p className="hero__copy">
            A static readout of {formatNumber(data.headline.denominators.all)} source samples, their candidate funnel,
            rejection modes, extraction origins, and selected traceable spot checks.
          </p>
        </div>
        <nav aria-label="Analysis sections">
          <a href="#outcomes">Outcomes</a>
          <a href="#origins">Origins</a>
          {data.candidate_evaluation && <a href="#evaluation">Candidate tests</a>}
          {data.failure_browser && <a href="#failures">Failures</a>}
          <a href="#spot-checks">Spot checks</a>
        </nav>
      </header>

      <section className="metric-grid" aria-label="Headline findings">
        <article className="metric metric--accent">
          <span>Function candidates extracted</span>
          <strong>{formatPercent(extracted?.rate_of_all ?? 0)}</strong>
          <small>{formatNumber(extracted?.count_all ?? 0)} of all samples</small>
        </article>
        <article className="metric">
          <span>Missing decoder output</span>
          <strong>{formatPercent(missing?.rate_of_all ?? 0)}</strong>
          <small>{formatNumber(missing?.count_all ?? 0)} samples stop before preprocessing</small>
        </article>
        <article className="metric">
          <span>Samples with no final candidate</span>
          <strong>{formatPercent(noCandidates?.rate ?? 0)}</strong>
          <small>{formatNumber(noCandidates?.sample_count ?? 0)} across all source samples</small>
        </article>
        <article className="metric">
          <span>Validated candidate rows</span>
          <strong>{formatNumber(data.headline.candidate_invariants.successful_candidate_rows)}</strong>
          <small>matches the final-candidate-row invariant</small>
        </article>
      </section>

      <section className="panel funnel-panel" aria-labelledby="funnel-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">funnel</p>
            <h2 id="funnel-title">Most nonblank responses yield a function; the visible loss starts earlier.</h2>
          </div>
          <span className="source-note">schema v{data.schema_version} · static snapshot</span>
        </div>
        <div className="funnel">
          {data.headline.funnel.map((stage, index) => (
            <article className="funnel__stage" key={stage.stage}>
              <span className="funnel__index">{String(index + 1).padStart(2, "0")}</span>
              <h3>{stage.stage}</h3>
              <strong>{formatNumber(stage.count)}</strong>
              <span>{formatPercent(stage.rate)} · {stage.rate_label}</span>
              <div aria-hidden="true"><i style={{ width: `${Math.min(stage.rate * 100, 100)}%` }} /></div>
            </article>
          ))}
        </div>
      </section>

      <section className="two-column" id="outcomes" aria-label="Outcome and failure analysis">
        <article className="panel">
          <div className="section-heading compact">
            <div>
              <p className="eyebrow">outcomes</p>
              <h2>Where samples landed</h2>
            </div>
            <span>share of all samples</span>
          </div>
          <div className="outcome-list">
            {data.headline.outcomes.map((outcome) => (
              <div className="outcome-row" key={outcome.outcome}>
                <OutcomeBadge outcome={outcome.outcome} />
                <strong>{formatNumber(outcome.count_all)}</strong>
                <span>{formatPercent(outcome.rate_of_all)}</span>
                <i aria-hidden="true"><b style={{ width: `${outcome.rate_of_all * 100}%` }} /></i>
              </div>
            ))}
          </div>
        </article>
        <article className="panel">
          <div className="section-heading compact">
            <div>
              <p className="eyebrow">failure modes</p>
              <h2>Candidate-level rejections are concentrated in compilation.</h2>
            </div>
          </div>
          <div className="table-scroll">
            <table>
              <thead><tr><th>Step</th><th>Reason</th><th>Scope</th><th>Rows</th><th>Samples</th></tr></thead>
              <tbody>
                {data.failure_modes.map((failure) => (
                  <tr key={`${failure.scope}-${failure.failed_step}-${failure.reason}`}>
                    <td>{humanize(failure.failed_step)}</td><td>{humanize(failure.reason)}</td><td>{humanize(failure.scope)}</td>
                    <td>{formatNumber(failure.count)}</td><td>{formatNumber(failure.sample_count)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>
      </section>

      <section className="two-column" id="origins" aria-label="Origin and multiplicity analysis">
        <article className="panel">
          <div className="section-heading compact">
            <div><p className="eyebrow">origins</p><h2>Fenced blocks and markdown wrappers account for nearly all converged candidates.</h2></div>
          </div>
          <div className="table-scroll">
            <table><thead><tr><th>Strategy / variant</th><th>Extracted</th><th>Final origins</th><th>Recovery</th><th>Convergence</th></tr></thead>
              <tbody>{data.origin_contribution.map((origin) => <tr key={`${origin.strategy}-${origin.variant}`}>
                <td><strong>{humanize(origin.strategy)}</strong><small>{humanize(origin.variant)}</small></td><td>{formatNumber(origin.extracted_candidate_count)}</td><td>{formatNumber(origin.final_candidate_origin_count)}</td><td>{formatPercent(origin.recovery_rate)}</td><td>{formatPercent(origin.convergence_rate)}</td>
              </tr>)}</tbody>
            </table>
          </div>
        </article>
        <article className="panel">
          <div className="section-heading compact">
            <div><p className="eyebrow">multiplicity</p><h2>One final candidate is the usual outcome.</h2></div>
            <label className="select-label">Denominator
              <select value={multiplicityDenominator} onChange={(event) => setMultiplicityDenominator(event.target.value as Denominator)}>
                <option value="all">all samples</option><option value="present">output present</option><option value="nonblank">output nonblank</option>
              </select>
            </label>
          </div>
          <div className="multiplicity-list">
            {multiplicity.map((row) => <div key={row.final_candidate_count}>
              <span>{row.final_candidate_count === 8 ? "8 candidates" : `${row.final_candidate_count} ${row.final_candidate_count === 1 ? "candidate" : "candidates"}`}</span>
              <i aria-hidden="true"><b style={{ width: `${row.rate * 100}%` }} /></i>
              <strong>{formatPercent(row.rate)}</strong><small>{formatNumber(row.sample_count)}</small>
            </div>)}
          </div>
        </article>
      </section>

      <section className="panel crosstab" aria-labelledby="crosstab-title">
        <div className="section-heading">
          <div><p className="eyebrow">crosstab</p><h2 id="crosstab-title">Compare an outcome across source dimensions.</h2></div>
          <div className="filters">
            <label>Dimension<select value={crosstabDimension} onChange={(event) => setCrosstabDimension(event.target.value)}>{dimensions.map((dimension) => <option key={dimension} value={dimension}>{humanize(dimension)}</option>)}</select></label>
            <label>Crosstab outcome<select value={crosstabOutcome} onChange={(event) => setCrosstabOutcome(event.target.value)}>{outcomeNames.map((outcome) => <option key={outcome} value={outcome}>{humanize(outcome)}</option>)}</select></label>
            <label>Denominator<select value={crosstabDenominator} onChange={(event) => setCrosstabDenominator(event.target.value as Denominator)}><option value="all">all</option><option value="present">present</option><option value="nonblank">nonblank</option></select></label>
          </div>
        </div>
        <div className="table-scroll"><table><thead><tr><th>{humanize(crosstabDimension)}</th><th>Outcome count</th><th>Denominator</th><th>Rate</th></tr></thead>
          <tbody>{crosstab.map((row) => <tr key={row.value}><td>{row.value}</td><td>{formatNumber(row.count)}</td><td>{formatNumber(row.denominator_count)}</td><td>{formatPercent(row.rate)}</td></tr>)}</tbody>
        </table></div>
      </section>

      {data.candidate_evaluation ? (
        <EvaluationAnalysis evaluation={data.candidate_evaluation} />
      ) : (
        <EvaluationUnavailable />
      )}

      {data.failure_browser && (
        <FailureBrowser
          browser={data.failure_browser}
          loader={failureLoader}
          renderExample={(example) => (
            <SpotCheck
              example={example}
              eyebrow="Active failure"
              titleId="failure-detail-title"
            />
          )}
        />
      )}

      <section className="spot-checks" id="spot-checks" aria-labelledby="spot-checks-title">
        <div className="section-heading">
          <div><p className="eyebrow">spot checks</p><h2 id="spot-checks-title">Twelve diagnostic examples make the aggregate story inspectable.</h2></div>
          <div className="filters filters--examples">
            <label>Search examples<input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="sample ID, source, task…" /></label>
            <label>Spot-check outcome<select value={exampleOutcome} onChange={(event) => setExampleOutcome(event.target.value)}><option value="all">all outcomes</option>{outcomeNames.map((outcome) => <option key={outcome} value={outcome}>{humanize(outcome)}</option>)}</select></label>
          </div>
        </div>
        <div className="spot-checks__layout">
          <div className="example-list" aria-label="Filtered examples">
            {examples.map((example) => <button className={selectedExample?.sample_id === example.sample_id ? "example-card example-card--selected" : "example-card"} key={example.sample_id} onClick={() => setSelectedId(example.sample_id)} type="button">
              <OutcomeBadge outcome={example.outcome} /><strong>{example.sample_id.slice(0, 16)}…</strong><span>{example.context.task_id ?? example.context.source_kind ?? "metadata unavailable"}</span><CandidateCount count={example.final_candidate_count} />
            </button>)}
            {examples.length === 0 && <p className="empty-copy">No example matches those filters.</p>}
          </div>
          {selectedExample && <SpotCheck example={selectedExample} />}
        </div>
      </section>
    </main>
  );
}
