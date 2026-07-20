import { useMemo, useState } from "react";

import { CodeBlock, StatusBadge, type StatusBadgeStatus } from "@dr-code/viewer";

import {
  filterEvaluationExamples,
  type CandidateEvaluation,
  type EvaluationComparison,
  type EvaluationExample,
  type TestOutcome,
} from "./data";

const TEST_OUTCOMES: TestOutcome[] = [
  "passed",
  "failed",
  "timed_out",
  "infrastructure_failure",
];
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

function testOutcomeStatus(outcome: TestOutcome): StatusBadgeStatus {
  if (outcome === "passed") return "success";
  if (outcome === "failed") return "failure";
  if (outcome === "timed_out") return "warning";
  return "neutral";
}

function TestOutcomeBadge({ outcome }: { outcome: TestOutcome }) {
  return (
    <StatusBadge status={testOutcomeStatus(outcome)}>
      {outcome === "infrastructure_failure" ? "infrastructure" : humanize(outcome)}
    </StatusBadge>
  );
}

function diagnosticValue(value: boolean | number | string | null): string {
  if (value === null) return "not recorded";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") return formatNumber(value);
  return value;
}

function exampleIdentity(example: EvaluationExample): string {
  return `${example.sample_id}:${example.candidate_index}:${example.candidate_id}`;
}

function abbreviatedHash(value: string): string {
  if (value.length <= 24) return value;
  return `${value.slice(0, 12)}…${value.slice(-8)}`;
}

function EvaluationContext({ evaluation }: { evaluation: CandidateEvaluation }) {
  const coordinates = evaluation.summary.provenance?.semantic_coordinates;
  const environment = coordinates?.sandbox_image || coordinates?.runner_identity;
  const context = [
    coordinates?.metrics_profile
      ? { label: "Metrics profile", title: coordinates.metrics_profile, value: coordinates.metrics_profile }
      : null,
    coordinates?.operator
      ? { label: "Operator", title: coordinates.operator, value: coordinates.operator }
      : null,
    environment
      ? {
          label: coordinates?.sandbox_image ? "Sandbox image" : "Runner identity",
          title: environment,
          value: environment,
        }
      : null,
    coordinates?.snapshot_sha256
      ? {
          label: "Snapshot",
          title: coordinates.snapshot_sha256,
          value: abbreviatedHash(coordinates.snapshot_sha256),
        }
      : null,
  ].filter((item) => item !== null);
  const limitations = evaluation.summary.limitations ?? [];

  if (context.length === 0 && limitations.length === 0) return null;

  return (
    <aside className="evaluation-context" aria-label="Evaluation provenance and limitations">
      {context.length > 0 && (
        <dl>
          {context.map((item) => (
            <div key={item.label}>
              <dt>{item.label}</dt>
              <dd><code title={item.title}>{item.value}</code></dd>
            </div>
          ))}
        </dl>
      )}
      {limitations.length > 0 && (
        <div className="evaluation-limitations">
          <strong>Evaluation limitations</strong>
          <ul>{limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul>
        </div>
      )}
    </aside>
  );
}

function TestExampleDetail({ example }: { example: EvaluationExample }) {
  const diagnostics = Object.entries(example.diagnostics).filter(
    ([key]) => key !== "failure_message",
  );

  return (
    <article className="spot-check-detail test-example-detail" aria-labelledby="test-spot-check-title">
      <div className="spot-check-detail__heading">
        <div>
          <p className="eyebrow">Active candidate test</p>
          <h3 id="test-spot-check-title">{example.candidate_id}</h3>
          <p className="identifier-line">sample {example.sample_id} · candidate {example.candidate_index + 1}</p>
        </div>
        <div className="badge-row">
          <TestOutcomeBadge outcome={example.test_outcome} />
          <StatusBadge status="neutral">{humanize(example.record_status)}</StatusBadge>
        </div>
      </div>

      <dl className="metadata-grid">
        <div><dt>task</dt><dd>{example.task_id}</dd></div>
        <div><dt>official outcome</dt><dd>{humanize(example.official_outcome ?? "not recorded")}</dd></div>
        <div><dt>evaluation key</dt><dd>{example.evaluation_key}</dd></div>
        {Object.entries(example.context).map(([key, value]) => (
          <div key={key}><dt>{humanize(key)}</dt><dd>{value ?? "not recorded"}</dd></div>
        ))}
      </dl>

      <div className="category-row" aria-label="Candidate extraction origins">
        {example.origins.length > 0 ? example.origins.map(({ strategy, variant }) => (
          <span key={`${strategy}-${variant}`}>{humanize(strategy)} / {humanize(variant)}</span>
        )) : <span>no extraction origin recorded</span>}
      </div>

      <div className="test-example-grid">
        <section aria-labelledby="tested-source-title">
          <h4 id="tested-source-title">Tested candidate</h4>
          <CodeBlock code={example.cleaned_source} lang="python" className="analysis-code" />
        </section>
        <section aria-labelledby="test-diagnostics-title">
          <h4 id="test-diagnostics-title">Execution diagnostics</h4>
          <dl className="test-diagnostics">
            {diagnostics.map(([key, value]) => (
              <div key={key}>
                <dt>{humanize(key)}</dt>
                <dd>{diagnosticValue(value)}</dd>
              </div>
            ))}
          </dl>
          {example.diagnostics.failure_message && (
            <div className="failure-message">
              <strong>Failure message</strong>
              <p>{example.diagnostics.failure_message}</p>
            </div>
          )}
        </section>
      </div>
    </article>
  );
}

interface ComparisonDisplayRow {
  allRate: number | null;
  evaluatedCount: number;
  evaluatedRate: number | null;
  failedCount: number;
  infrastructureCount: number;
  label: string;
  notExtractedCount: number | null;
  passedCount: number;
  sampleCount: number;
  timedOutCount: number;
}

function comparisonRow(row: EvaluationComparison): ComparisonDisplayRow {
  return {
    allRate: row.pass_rate_of_all_samples,
    evaluatedCount: row.evaluated_sample_count,
    evaluatedRate: row.pass_rate_of_evaluated_samples,
    failedCount: row.failed_count,
    infrastructureCount: row.infrastructure_failure_count,
    label: row.value,
    notExtractedCount: row.not_extracted_count ?? null,
    passedCount: row.passed_count,
    sampleCount: row.sample_count,
    timedOutCount: row.timed_out_count,
  };
}

function ComparisonTable({ evaluation }: { evaluation: CandidateEvaluation }) {
  const dimensions = Array.from(
    new Set(evaluation.test_success_by_dimension.map((row) => row.dimension)),
  ).sort();
  const [lens, setLens] = useState("origin");

  const rows: ComparisonDisplayRow[] = useMemo(() => {
    if (lens === "origin") {
      return evaluation.test_success_by_origin.map((row) => ({
        allRate: row.pass_rate,
        evaluatedCount: row.candidate_origin_count,
        evaluatedRate: row.pass_rate,
        failedCount: row.failed_count,
        infrastructureCount: row.infrastructure_failure_count,
        label: `${humanize(row.strategy)} / ${humanize(row.variant)}`,
        notExtractedCount: null,
        passedCount: row.passed_count,
        sampleCount: row.candidate_origin_count,
        timedOutCount: row.timed_out_count,
      }));
    }

    const source = lens === "multiplicity"
      ? evaluation.test_success_by_multiplicity
      : lens === "preprocessing_outcome"
        ? evaluation.test_success_by_preprocessing_outcome
        : evaluation.test_success_by_dimension.filter(
            (row) => row.dimension === lens.slice("dimension:".length),
          );
    return source.map(comparisonRow);
  }, [evaluation, lens]);

  const sortedRows = [...rows].sort(
    (left, right) => right.sampleCount - left.sampleCount || left.label.localeCompare(right.label),
  );
  const isOrigin = lens === "origin";
  const isMultiplicity = lens === "multiplicity";
  const denominatorNote = isOrigin
    ? "Counts are final-candidate origin attributions; pass rate uses attributed candidates."
    : isMultiplicity
      ? "Multiplicity rows include only samples with an extracted candidate; pass rate uses those evaluated samples."
      : "Pass / all includes samples with no extracted candidate; pass / evaluated starts after extraction.";

  return (
    <article className="panel evaluation-comparison">
      <div className="section-heading compact">
        <div>
          <p className="eyebrow">comparisons</p>
          <h2>Which preprocessing paths lead to passing code?</h2>
        </div>
        <label className="select-label">Comparison lens
          <select value={lens} onChange={(event) => setLens(event.target.value)}>
            <option value="origin">candidate extraction origin</option>
            <option value="multiplicity">candidate multiplicity</option>
            <option value="preprocessing_outcome">preprocessing outcome</option>
            {dimensions.map((dimension) => (
              <option key={dimension} value={`dimension:${dimension}`}>
                {humanize(dimension)}
              </option>
            ))}
          </select>
        </label>
      </div>
      <p className="panel-note">{denominatorNote}</p>
      <div className="table-scroll comparison-table">
        <table>
          <thead><tr>
            <th>Segment</th>
            <th>{isOrigin ? "Attributions" : isMultiplicity ? "Extracted samples" : "All samples"}</th>
            {!isOrigin && !isMultiplicity && <th>Evaluated samples</th>}
            <th>Passed</th><th>Failed</th><th>Timed out</th><th>Infrastructure</th>
            {!isOrigin && !isMultiplicity && <th>Not extracted</th>}
            <th>{isOrigin || isMultiplicity ? "Pass rate" : "Pass / evaluated"}</th>
            {!isOrigin && !isMultiplicity && <th>Pass / all</th>}
          </tr></thead>
          <tbody>
            {sortedRows.map((row) => (
              <tr key={row.label}>
                <td>{row.label}</td>
                <td>{formatNumber(row.sampleCount)}</td>
                {!isOrigin && !isMultiplicity && <td>{formatNumber(row.evaluatedCount)}</td>}
                <td>{formatNumber(row.passedCount)}</td>
                <td>{formatNumber(row.failedCount)}</td>
                <td>{formatNumber(row.timedOutCount)}</td>
                <td>{formatNumber(row.infrastructureCount)}</td>
                {!isOrigin && !isMultiplicity && <td>{row.notExtractedCount === null ? "—" : formatNumber(row.notExtractedCount)}</td>}
                <td>{formatPercent(row.evaluatedRate)}</td>
                {!isOrigin && !isMultiplicity && <td>{formatPercent(row.allRate)}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  );
}

export function EvaluationAnalysis({ evaluation }: { evaluation: CandidateEvaluation }) {
  const [search, setSearch] = useState("");
  const [outcome, setOutcome] = useState<TestOutcome | "all">("all");
  const [selectedId, setSelectedId] = useState(
    evaluation.examples[0] ? exampleIdentity(evaluation.examples[0]) : "",
  );

  const candidateCounts = new Map<TestOutcome, number>(
    TEST_OUTCOMES.map((testOutcome) => [testOutcome, 0]),
  );
  for (const row of evaluation.summary.candidate_outcomes) {
    candidateCounts.set(
      row.test_outcome,
      (candidateCounts.get(row.test_outcome) ?? 0) + row.candidate_count,
    );
  }
  const sampleRows = new Map(
    evaluation.summary.sample_best_outcomes.map((row) => [row.best_test_outcome, row]),
  );
  const candidateTotal = evaluation.summary.candidate_membership_count;
  const candidatePassed = candidateCounts.get("passed") ?? 0;
  const samplePassed = sampleRows.get("passed")?.sample_count ?? 0;
  const samplePassRate = sampleRows.get("passed")?.rate_of_extracted_samples ?? null;
  const interrupted = (candidateCounts.get("timed_out") ?? 0) +
    (candidateCounts.get("infrastructure_failure") ?? 0);
  const examples = useMemo(
    () => filterEvaluationExamples(evaluation.examples, search, outcome),
    [evaluation.examples, outcome, search],
  );
  const selected = examples.find((example) => exampleIdentity(example) === selectedId) ?? examples[0];

  return (
    <section className="evaluation" id="evaluation" aria-labelledby="evaluation-title">
      <div className="evaluation-hero">
        <div>
          <p className="eyebrow">candidate evaluation</p>
          <h2 id="evaluation-title">Preprocessing success is the starting line. Execution decides whether the code works.</h2>
        </div>
        <p>
          {formatNumber(samplePassed)} of {formatNumber(evaluation.summary.extracted_sample_count)} extracted samples have at least one passing candidate.
        </p>
      </div>

      <EvaluationContext evaluation={evaluation} />

      <div className="metric-grid evaluation-metrics" aria-label="Candidate evaluation findings">
        <article className="metric metric--accent">
          <span>Candidate pass rate</span>
          <strong>{formatPercent(candidateTotal === 0 ? null : candidatePassed / candidateTotal)}</strong>
          <small>{formatNumber(candidatePassed)} of {formatNumber(candidateTotal)} candidate memberships</small>
        </article>
        <article className="metric">
          <span>Sample best-of pass rate</span>
          <strong>{formatPercent(samplePassRate)}</strong>
          <small>{formatNumber(samplePassed)} extracted samples with ≥1 passing candidate</small>
        </article>
        <article className="metric">
          <span>Timeout or infrastructure</span>
          <strong>{formatPercent(candidateTotal === 0 ? null : interrupted / candidateTotal)}</strong>
          <small>{formatNumber(interrupted)} candidate outcomes need operational context</small>
        </article>
        <article className="metric">
          <span>Deduplicated executions</span>
          <strong>{formatNumber(evaluation.summary.deduplicated_evaluation_count)}</strong>
          <small>reused across {formatNumber(candidateTotal)} candidate memberships</small>
        </article>
      </div>

      <article className="panel evaluation-funnel-panel">
        <div className="section-heading">
          <div><p className="eyebrow">evaluation funnel</p><h2>Every final candidate is accounted for at the test boundary.</h2></div>
          <span>Outcome stages partition candidate memberships</span>
        </div>
        <div className="evaluation-funnel">
          {evaluation.summary.funnel.map((stage) => (
            <div className={`evaluation-funnel__stage evaluation-funnel__stage--${stage.stage.replaceAll(" ", "-")}`} key={stage.stage}>
              <span>{humanize(stage.stage)}</span>
              <strong>{formatNumber(stage.count)}</strong>
              <small>{formatPercent(stage.rate)}</small>
              <i aria-hidden="true"><b style={{ width: `${(stage.rate ?? 0) * 100}%` }} /></i>
            </div>
          ))}
        </div>
      </article>

      <div className="two-column evaluation-outcomes">
        <article className="panel">
          <div className="section-heading compact"><div><p className="eyebrow">candidate outcomes</p><h2>Individual candidate test results</h2></div><span>share of candidate memberships</span></div>
          <div className="test-outcome-list">
            {TEST_OUTCOMES.map((testOutcome) => {
              const count = candidateCounts.get(testOutcome) ?? 0;
              return <div key={testOutcome}><TestOutcomeBadge outcome={testOutcome} /><strong>{formatNumber(count)}</strong><span>{formatPercent(candidateTotal === 0 ? null : count / candidateTotal)}</span></div>;
            })}
          </div>
          <details className="outcome-details">
            <summary>Official outcome and failure-type detail</summary>
            <div className="table-scroll"><table><thead><tr><th>Category</th><th>Official outcome</th><th>Record status</th><th>Failure type</th><th>Candidates</th></tr></thead><tbody>
              {evaluation.summary.candidate_outcomes.map((row, index) => <tr key={`${row.test_outcome}-${row.official_outcome}-${row.failure_type}-${index}`}><td>{humanize(row.test_outcome)}</td><td>{humanize(row.official_outcome)}</td><td>{humanize(row.record_status)}</td><td>{humanize(row.failure_type)}</td><td>{formatNumber(row.candidate_count)}</td></tr>)}
            </tbody></table></div>
          </details>
        </article>
        <article className="panel">
          <div className="section-heading compact"><div><p className="eyebrow">sample best-of</p><h2>The strongest candidate determines each sample outcome.</h2></div><span>share of extracted samples</span></div>
          <div className="test-outcome-list">
            {TEST_OUTCOMES.map((testOutcome) => {
              const row = sampleRows.get(testOutcome);
              return <div key={testOutcome}><TestOutcomeBadge outcome={testOutcome} /><strong>{formatNumber(row?.sample_count ?? 0)}</strong><span>{formatPercent(row?.rate_of_extracted_samples ?? null)}</span></div>;
            })}
          </div>
          <p className="panel-note">Priority is passing, then failed, timed out, and infrastructure failure. A sample passes if any final candidate passes.</p>
        </article>
      </div>

      <ComparisonTable evaluation={evaluation} />

      <section className="test-examples" aria-labelledby="test-examples-title">
        <div className="section-heading">
          <div><p className="eyebrow">test-outcome spot checks</p><h2 id="test-examples-title">Inspect the candidate, execution result, and failure evidence together.</h2></div>
          <div className="filters filters--examples">
            <label>Search test examples<input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="task, sample, model, failure…" /></label>
            <label>Test outcome<select value={outcome} onChange={(event) => setOutcome(event.target.value as TestOutcome | "all")}><option value="all">all test outcomes</option>{TEST_OUTCOMES.map((testOutcome) => <option key={testOutcome} value={testOutcome}>{humanize(testOutcome)}</option>)}</select></label>
          </div>
        </div>
        <div className="spot-checks__layout">
          <div className="example-list" aria-label="Filtered candidate test examples">
            {examples.map((example) => (
              <button className={selected && exampleIdentity(selected) === exampleIdentity(example) ? "example-card test-example-card example-card--selected" : "example-card test-example-card"} key={exampleIdentity(example)} onClick={() => setSelectedId(exampleIdentity(example))} type="button">
                <TestOutcomeBadge outcome={example.test_outcome} />
                <strong>{example.task_id}</strong>
                <span>{example.candidate_id}</span>
                <span>{example.context.model ?? example.context.source_kind ?? "metadata unavailable"}</span>
              </button>
            ))}
            {examples.length === 0 && <p className="empty-copy">No candidate test example matches those filters.</p>}
          </div>
          {selected && <TestExampleDetail example={selected} />}
        </div>
      </section>
    </section>
  );
}

export function EvaluationUnavailable() {
  return (
    <section className="evaluation-unavailable" aria-labelledby="evaluation-unavailable-title">
      <StatusBadge status="neutral">preprocessing only</StatusBadge>
      <div>
        <h2 id="evaluation-unavailable-title">Candidate execution results are not part of this snapshot yet.</h2>
        <p>The preprocessing findings remain valid. Regenerate the authoritative analysis with candidate membership and result relations to unlock pass, failure, timeout, infrastructure, and best-of comparisons.</p>
      </div>
    </section>
  );
}
