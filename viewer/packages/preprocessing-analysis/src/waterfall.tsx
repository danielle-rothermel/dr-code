import { useEffect, useState } from "react";

import type { PreprocessingApi, WaterfallResponse } from "./api";
import { ExamplesPanel, type ExampleSelection } from "./examples-panel";
import { errorMessage, formatNumber, formatPercent } from "./format";

export function Waterfall({ api, runId }: { api: PreprocessingApi; runId: string }) {
  const [waterfall, setWaterfall] = useState<WaterfallResponse | null>(null);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const [selection, setSelection] = useState<ExampleSelection | null>(null);

  useEffect(() => {
    let active = true;
    setWaterfall(null);
    setError("");
    setSelection(null);
    void api.getWaterfall(runId).then(
      (response) => { if (active) setWaterfall(response); },
      (reason: unknown) => { if (active) setError(errorMessage(reason)); },
    );
    return () => { active = false; };
  }, [api, retry, runId]);

  return (
    <section className="surface" aria-labelledby="waterfall-title">
      <div className="surface-heading">
        <div><p className="eyebrow">Waterfall</p><h2 id="waterfall-title">Trace every stage back to examples</h2></div>
      </div>
      <p className="surface-copy">Counts progress from corpus rows through extraction and, when present, candidate evaluation. Activate any count to inspect its exact members.</p>

      {waterfall === null && error === "" && <p className="loading-state" role="status">Loading waterfall…</p>}
      {error !== "" && (
        <div className="error-state" role="alert"><strong>Could not load this run.</strong><span>{error}</span><button onClick={() => setRetry((value) => value + 1)} type="button">Retry</button></div>
      )}
      {waterfall !== null && waterfall.stages.length === 0 && <p className="empty-state">This run has no waterfall stages.</p>}
      {waterfall !== null && waterfall.stages.length > 0 && (
        <ol className="waterfall-list">
          {waterfall.stages.map((stage, index, stages) => {
            const previous = stages[index - 1];
            const lost = previous === undefined || previous.unit !== stage.unit
              ? null
              : Math.max(0, previous.count - stage.count);
            return (
              <li key={stage.id}>
                <div className="waterfall-card">
                  <div className="stage-index" aria-hidden="true">{String(index + 1).padStart(2, "0")}</div>
                  <div className="stage-copy"><h3>{stage.label}</h3><span>{stage.unit}</span></div>
                  <button
                    aria-label={`Inspect ${formatNumber(stage.count)} examples at ${stage.label}`}
                    className="count-button"
                    onClick={() => setSelection({ query: { stage_id: stage.id }, runId, title: stage.label })}
                    type="button"
                  >{formatNumber(stage.count)}</button>
                  <div className="stage-rate"><strong>{formatPercent(stage.rate)}</strong><small>of {formatNumber(stage.denominator_count)}</small></div>
                  <div className="stage-bar" aria-hidden="true"><i style={{ width: `${Math.min(100, Math.max(0, (stage.rate ?? 0) * 100))}%` }} /></div>
                </div>
                {lost !== null && lost > 0 && (
                  <button
                    aria-label={`Inspect ${formatNumber(lost)} examples that did not reach ${stage.label}`}
                    className="loss-button"
                    onClick={() => setSelection({
                      query: { stage_id: `lost:${stage.id}` },
                      runId,
                      title: `Did not reach ${stage.label}`,
                    })}
                    type="button"
                  >↓ {formatNumber(lost)} did not advance</button>
                )}
              </li>
            );
          })}
        </ol>
      )}

      {selection !== null && <ExamplesPanel api={api} selection={selection} />}
    </section>
  );
}
