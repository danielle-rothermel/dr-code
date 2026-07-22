import { useEffect, useState } from "react";

import type { ExampleDetail, ExampleQuery, ExampleSummary, PreprocessingApi } from "./api";
import { ExampleDetail as ExampleDetailView } from "./example-detail";
import { errorMessage, formatNumber, humanize } from "./format";

const PAGE_SIZE = 25;

type DetailState =
  | { status: "loading" }
  | { data: ExampleDetail; status: "success" }
  | { error: string; status: "error" };

export interface ExampleSelection {
  query: ExampleQuery;
  runId: string;
  title: string;
}

export function ExamplesPanel({
  api,
  selection,
}: {
  api: PreprocessingApi;
  selection: ExampleSelection;
}) {
  const [examples, setExamples] = useState<ExampleSummary[] | null>(null);
  const [examplesError, setExamplesError] = useState("");
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [selectedId, setSelectedId] = useState("");
  const [baselineDetail, setBaselineDetail] = useState<DetailState | null>(null);
  const [candidateDetail, setCandidateDetail] = useState<DetailState | null>(null);

  const queryKey = JSON.stringify(selection.query);
  const compareRunId = selection.query.compare_run_id;

  useEffect(() => {
    setOffset(0);
  }, [queryKey]);

  useEffect(() => {
    let active = true;
    setExamples(null);
    setExamplesError("");
    void api.getExamples(selection.runId, { ...selection.query, limit: PAGE_SIZE, offset }).then(
      (response) => {
        if (!active) return;
        setExamples(response.items);
        setTotal(response.total);
        setSelectedId((current) => response.items.some(({ sample_id }) => sample_id === current)
          ? current
          : response.items[0]?.sample_id ?? "");
      },
      (error: unknown) => {
        if (active) setExamplesError(errorMessage(error));
      },
    );
    return () => { active = false; };
  }, [api, offset, queryKey, selection.query, selection.runId]);

  useEffect(() => {
    if (selectedId === "") {
      setBaselineDetail(null);
      setCandidateDetail(null);
      return;
    }
    let active = true;
    setBaselineDetail({ status: "loading" });
    void api.getExample(selection.runId, selectedId).then(
      (value) => { if (active) setBaselineDetail({ data: value, status: "success" }); },
      (error: unknown) => { if (active) setBaselineDetail({ error: errorMessage(error), status: "error" }); },
    );
    if (compareRunId === undefined) {
      setCandidateDetail(null);
    } else {
      setCandidateDetail({ status: "loading" });
      void api.getExample(compareRunId, selectedId).then(
        (value) => { if (active) setCandidateDetail({ data: value, status: "success" }); },
        (error: unknown) => { if (active) setCandidateDetail({ error: errorMessage(error), status: "error" }); },
      );
    }
    return () => { active = false; };
  }, [api, compareRunId, selectedId, selection.runId]);

  function renderDetail(state: DetailState | null, side?: "Baseline" | "Candidate") {
    if (state === null || state.status === "loading") {
      return <p className="loading-state" role="status">Loading {side?.toLowerCase() ?? "example"} detail…</p>;
    }
    if (state.status === "error") {
      return <p className="inline-error" role="alert">{state.error}</p>;
    }
    return (
      <ExampleDetailView
        example={state.data}
        eyebrow={side === undefined ? undefined : `${side} detail`}
        titleId={side === undefined ? undefined : `${side.toLowerCase()}-example-detail-title`}
      />
    );
  }

  return (
    <section className="drilldown" aria-labelledby="drilldown-title">
      <div className="surface-heading compact">
        <div><p className="eyebrow">Inspect count</p><h3 id="drilldown-title">{selection.title}</h3></div>
        <span>{formatNumber(total)} examples</span>
      </div>
      {examples === null && examplesError === "" && <p className="loading-state" role="status">Loading examples…</p>}
      {examplesError !== "" && <p className="inline-error" role="alert">{examplesError}</p>}
      {examples !== null && examples.length === 0 && <p className="empty-state">No examples belong to this count.</p>}
      {examples !== null && examples.length > 0 && (
        <div className="drilldown-layout">
          <div>
            <div className="example-list">
              {examples.map((example) => (
                <button
                  aria-current={example.sample_id === selectedId ? "true" : undefined}
                  key={example.sample_id}
                  onClick={() => setSelectedId(example.sample_id)}
                  type="button"
                >
                  <strong>{example.sample_id}</strong>
                  <span>{humanize(example.outcome)}</span>
                  <small>{String(example.context.task_id ?? example.raw_preview ?? "No preview")}</small>
                </button>
              ))}
            </div>
            <div className="pagination" aria-label="Drill-down pagination">
              <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))} type="button">Previous</button>
              <span>{offset + 1}–{Math.min(offset + PAGE_SIZE, total)} of {total}</span>
              <button disabled={offset + PAGE_SIZE >= total} onClick={() => setOffset(offset + PAGE_SIZE)} type="button">Next</button>
            </div>
          </div>
          <div>
            {compareRunId === undefined ? renderDetail(baselineDetail) : (
              <div className="comparison-details">
                <section aria-label="Baseline example detail">{renderDetail(baselineDetail, "Baseline")}</section>
                <section aria-label="Candidate example detail">{renderDetail(candidateDetail, "Candidate")}</section>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
