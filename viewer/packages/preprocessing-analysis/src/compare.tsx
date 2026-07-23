import { useEffect, useState } from "react";

import type { CompareResponse, PreprocessingApi, RunSummary } from "./api";
import { ExamplesPanel, type ExampleSelection } from "./examples-panel";
import { errorMessage, formatDelta, formatNumber, formatPercent, formatRateDelta, humanize } from "./format";

export function Compare({
  api,
  baselineRun,
  candidateRun,
}: {
  api: PreprocessingApi;
  baselineRun: RunSummary;
  candidateRun: RunSummary;
}) {
  const baselineRunId = baselineRun.run_id;
  const candidateRunId = candidateRun.run_id;
  const [comparison, setComparison] = useState<CompareResponse | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const [selection, setSelection] = useState<ExampleSelection | null>(null);

  useEffect(() => {
    let active = true;
    setComparison(null);
    setError("");
    setSelection(null);
    void api.compare(baselineRunId, candidateRunId).then(
      (response) => { if (active) setComparison(response); },
      (reason: unknown) => { if (active) setError(errorMessage(reason)); },
    );
    return () => { active = false; };
  }, [api, baselineRunId, candidateRunId, retry]);

  const baselineOutcomes = comparison === null
    ? []
    : Array.from(new Set(comparison.transitions.map(({ baseline_outcome }) => baseline_outcome))).sort();
  const candidateOutcomes = comparison === null
    ? []
    : Array.from(new Set(comparison.transitions.map(({ candidate_outcome }) => candidate_outcome))).sort();

  return (
    <section className="surface" aria-labelledby="compare-title">
      <div className="surface-heading"><div><p className="eyebrow">Compare</p><h2 id="compare-title">Compatible before / after deltas</h2></div></div>
      <p className="surface-copy">Counts are corpus rows, and every percentage uses all corpus rows as its denominator. Incompatible corpora, stage mappings, or evaluation semantics are rejected by the service.</p>

      {comparison === null && error === "" && <p className="loading-state" role="status">Checking compatibility and loading comparison…</p>}
      {error !== "" && (
        <div className="error-state" role="alert">
          <strong>These runs cannot be compared.</strong><span>{error}</span>
          <button onClick={() => setRetry((value) => value + 1)} type="button">Try again</button>
        </div>
      )}
      {comparison !== null && (
        <>
          <div className="table-scroll">
            <table aria-label="Corpus row comparison" className="comparison-table">
              <thead>
                <tr>
                  <th>Stage</th>
                  <th>
                    Before
                    <small>{baselineRun.label} · {baselineRun.definition_id}@{String(baselineRun.semantic_coordinates.definition_version ?? "unknown")}</small>
                  </th>
                  <th>
                    After
                    <small>{candidateRun.label} · {candidateRun.definition_id}@{String(candidateRun.semantic_coordinates.definition_version ?? "unknown")}</small>
                  </th>
                  <th>Count Δ</th>
                  <th>Row share Δ</th>
                </tr>
              </thead>
              <tbody>
                {comparison.stages.map((stage) => (
                  <tr key={stage.id}>
                    <th scope="row">{stage.label}<small>corpus rows</small></th>
                    <td>
                      <button
                        aria-label={`Inspect ${formatNumber(stage.baseline_count)} baseline examples at ${stage.label}`}
                        className="table-count-button"
                        onClick={() => setSelection({ query: { stage_id: stage.id }, runId: baselineRunId, title: `${stage.label} · baseline` })}
                        type="button"
                      >{formatNumber(stage.baseline_count)}</button>
                      <small>{formatPercent(stage.baseline_rate)}</small>
                    </td>
                    <td>
                      <button
                        aria-label={`Inspect ${formatNumber(stage.candidate_count)} candidate examples at ${stage.label}`}
                        className="table-count-button"
                        onClick={() => setSelection({ query: { stage_id: stage.id }, runId: candidateRunId, title: `${stage.label} · candidate` })}
                        type="button"
                      >{formatNumber(stage.candidate_count)}</button>
                      <small>{formatPercent(stage.candidate_rate)}</small>
                    </td>
                    <td className={stage.count_delta > 0 ? "delta-positive" : stage.count_delta < 0 ? "delta-negative" : ""}>{formatDelta(stage.count_delta)}</td>
                    <td className={(stage.rate_delta ?? 0) > 0 ? "delta-positive" : (stage.rate_delta ?? 0) < 0 ? "delta-negative" : ""}>{formatRateDelta(stage.rate_delta)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="transition-section">
            <div className="surface-heading compact"><div><p className="eyebrow">Transitions</p><h3>Terminal outcome matrix</h3></div></div>
            {comparison.transitions.length === 0 ? <p className="empty-state">No terminal transitions were found.</p> : (
              <div className="table-scroll">
                <table aria-label="Terminal outcome transitions" className="transition-matrix">
                  <caption className="visually-hidden">Before outcomes by after outcome</caption>
                  <thead>
                    <tr>
                      <th scope="col">Before ↓ / After →</th>
                      {candidateOutcomes.map((outcome) => (
                        <th key={outcome} scope="col">{humanize(outcome)}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {baselineOutcomes.map((baselineOutcome) => (
                      <tr key={baselineOutcome}>
                        <th scope="row">{humanize(baselineOutcome)}</th>
                        {candidateOutcomes.map((candidateOutcome) => {
                          const transition = comparison.transitions.find(
                            ({ baseline_outcome, candidate_outcome }) => baseline_outcome === baselineOutcome && candidate_outcome === candidateOutcome,
                          );
                          return (
                            <td key={candidateOutcome}>
                              {transition === undefined ? <span className="matrix-zero">0</span> : (
                                <button
                                  aria-label={`${humanize(transition.baseline_outcome)} → ${humanize(transition.candidate_outcome)}: inspect ${formatNumber(transition.count)} examples`}
                                  onClick={() => setSelection({
                                    query: {
                                      baseline_outcome: transition.baseline_outcome,
                                      candidate_outcome: transition.candidate_outcome,
                                      compare_run_id: candidateRunId,
                                    },
                                    runId: baselineRunId,
                                    title: `${humanize(transition.baseline_outcome)} → ${humanize(transition.candidate_outcome)}`,
                                  })}
                                  type="button"
                                >
                                  {formatNumber(transition.count)}
                                </button>
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}

      {selection !== null && <ExamplesPanel api={api} selection={selection} />}
    </section>
  );
}
