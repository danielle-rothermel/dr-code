import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { type PreprocessingApi, type RunSummary, type Tag, defaultApi } from "./api";
import { Compare } from "./compare";
import { errorMessage } from "./format";
import { Review, type RegisterBeforeLeave } from "./review";
import { Waterfall } from "./waterfall";

type Surface = "waterfall" | "compare" | "review";

export function PreprocessingViewer({ api = defaultApi }: { api?: PreprocessingApi }) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [runsError, setRunsError] = useState("");
  const [tags, setTags] = useState<Tag[]>([]);
  const [tagsError, setTagsError] = useState("");
  const [selectedRunId, setSelectedRunId] = useState("");
  const [compareRunId, setCompareRunId] = useState("");
  const [surface, setSurface] = useState<Surface>("waterfall");
  const [retry, setRetry] = useState(0);
  const beforeLeaveRef = useRef<() => Promise<boolean>>(async () => true);
  const registerBeforeLeave = useCallback<RegisterBeforeLeave>((handler) => {
    beforeLeaveRef.current = handler;
    return () => {
      if (beforeLeaveRef.current === handler) beforeLeaveRef.current = async () => true;
    };
  }, []);

  async function navigate(action: () => void): Promise<boolean> {
    const canLeave = await beforeLeaveRef.current();
    if (canLeave) action();
    return canLeave;
  }

  useEffect(() => {
    let active = true;
    setRuns(null);
    setRunsError("");
    void api.getRuns().then(
      (availableRuns) => {
        if (!active) return;
        setRuns(availableRuns);
        setSelectedRunId((current) => availableRuns.some(({ run_id }) => run_id === current)
          ? current
          : availableRuns[0]?.run_id ?? "");
        setCompareRunId((current) => availableRuns.some(({ run_id }) => run_id === current)
          ? current
          : availableRuns[1]?.run_id ?? availableRuns[0]?.run_id ?? "");
      },
      (error: unknown) => { if (active) setRunsError(errorMessage(error)); },
    );
    return () => { active = false; };
  }, [api, retry]);

  useEffect(() => {
    let active = true;
    setTagsError("");
    void api.getTags().then(
      (availableTags) => { if (active) setTags(availableTags); },
      (error: unknown) => { if (active) setTagsError(errorMessage(error)); },
    );
    return () => { active = false; };
  }, [api, retry]);

  const selectedRun = useMemo(
    () => runs?.find(({ run_id }) => run_id === selectedRunId),
    [runs, selectedRunId],
  );
  const canCompare = (runs?.length ?? 0) >= 2;

  function addTag(tag: Tag) {
    setTags((current) => [...current.filter(({ tag_id }) => tag_id !== tag.tag_id), tag]
      .sort((left, right) => left.name.localeCompare(right.name)));
  }

  return (
    <main>
      <header className="app-header">
        <div className="app-title">
          <p className="eyebrow">dr-code · local analysis</p>
          <h1>Preprocessing viewer</h1>
          <p>Inspect run waterfalls, compare compatible changes, and persist failure review without copied corpus snapshots.</p>
        </div>
        {runs !== null && runs.length > 0 && (
          <div className="run-controls">
            <label htmlFor="active-run">Active run</label>
            <select
              id="active-run"
              onChange={(event) => {
                const nextRunId = event.target.value;
                const select = event.currentTarget;
                void navigate(() => {
                  setSelectedRunId(nextRunId);
                  if (nextRunId === compareRunId) {
                    setCompareRunId(runs.find(({ run_id }) => run_id !== nextRunId)?.run_id ?? nextRunId);
                  }
                }).then((didNavigate) => {
                  if (!didNavigate) select.value = selectedRunId;
                });
              }}
              value={selectedRunId}
            >
              {runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.label}</option>)}
            </select>
            {surface === "compare" && canCompare && (
              <>
                <label htmlFor="compare-run">Candidate run</label>
                <select id="compare-run" onChange={(event) => setCompareRunId(event.target.value)} value={compareRunId}>
                  {runs.filter(({ run_id }) => run_id !== selectedRunId).map((run) => (
                    <option key={run.run_id} value={run.run_id}>{run.label}</option>
                  ))}
                </select>
              </>
            )}
          </div>
        )}
      </header>

      {runs === null && runsError === "" && <p className="loading-state app-state" role="status">Loading registered runs…</p>}
      {runsError !== "" && (
        <div className="error-state app-state" role="alert"><strong>Could not connect to the preprocessing service.</strong><span>{runsError}</span><button onClick={() => setRetry((value) => value + 1)} type="button">Retry</button></div>
      )}
      {runs !== null && runs.length === 0 && (
        <section className="empty-state app-state"><h2>No runs are registered</h2><p>Restart the local viewer command with at least one named run descriptor.</p></section>
      )}

      {selectedRun !== undefined && (
        <>
          <nav className="app-nav" aria-label="Viewer sections">
            <button aria-current={surface === "waterfall" ? "page" : undefined} onClick={() => void navigate(() => setSurface("waterfall"))} type="button">Waterfall</button>
            {canCompare && <button aria-current={surface === "compare" ? "page" : undefined} onClick={() => void navigate(() => setSurface("compare"))} type="button">Compare</button>}
            <button aria-current={surface === "review" ? "page" : undefined} onClick={() => void navigate(() => setSurface("review"))} type="button">Review</button>
          </nav>

          <aside className="run-provenance" aria-label="Active run provenance">
            <strong>{selectedRun.label}</strong>
            <span>definition {selectedRun.definition_id}@{String(selectedRun.semantic_coordinates.definition_version ?? "unknown")}</span>
            <span title={selectedRun.corpus_sha256}>corpus {selectedRun.corpus_sha256.slice(0, 12)}…</span>
            {selectedRun.has_evaluation && <span>evaluation attached</span>}
          </aside>

          {tagsError !== "" && surface === "review" && <p className="inline-error" role="alert">Tags unavailable: {tagsError}</p>}
          {surface === "waterfall" && <Waterfall api={api} runId={selectedRunId} />}
          {surface === "compare" && canCompare && compareRunId !== selectedRunId && (
            <Compare api={api} baselineRunId={selectedRunId} candidateRunId={compareRunId} />
          )}
          {surface === "review" && <Review api={api} onTagCreated={addTag} registerBeforeLeave={registerBeforeLeave} runId={selectedRunId} tags={tags} />}
        </>
      )}
    </main>
  );
}
