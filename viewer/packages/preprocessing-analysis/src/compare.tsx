import { useEffect, useState } from "react";

import type { CompareResponse, PreprocessingApi } from "./api";
import { ExamplesPanel, type ExampleSelection } from "./examples-panel";
import { errorMessage, formatDelta, formatNumber, formatPercent, formatRateDelta, humanize } from "./format";

export function Compare({
  api,
  baselineRunId,
  candidateRunId,
}: {
  api: PreprocessingApi;
  baselineRunId: string;
  candidateRunId: string;
}) {
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

  return (
    <section className="surface" aria-labelledby="compare-title">
      <div className="surface-heading"><div><p className="eyebrow">Compare</p><h2 id="compare-title">Compatible before / after deltas</h2></div></div>
      <p className="surface-copy">Counts and rates share named stage contracts. Incompatible corpora, stage mappings, or evaluation semantics are rejected by the service.</p>

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
            <table className="comparison-table">
              <thead><tr><th>Stage</th><th>Baseline</th><th>Candidate</th><th>Count Δ</th><th>Rate Δ</th></tr></thead>
              <tbody>
                {comparison.stages.map((stage) => (
                  <tr key={stage.id}>
                    <th scope="row">{stage.label}<small>{stage.unit}</small></th>
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
              <div className="transition-grid" aria-label="Terminal outcome transitions">
                {comparison.transitions.map((transition) => (
                  <button
                    key={`${transition.baseline_outcome}-${transition.candidate_outcome}`}
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
                    <span>{humanize(transition.baseline_outcome)}</span><i aria-hidden="true">→</i><span>{humanize(transition.candidate_outcome)}</span><strong>{formatNumber(transition.count)}</strong>
                  </button>
                ))}
              </div>
            )}
          </div>
        </>
      )}

      {selection !== null && <ExamplesPanel api={api} selection={selection} />}
    </section>
  );
}
