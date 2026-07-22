import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { StatusBadge } from "@dr-code/viewer";

import type {
  Annotation,
  AnnotationIdentity,
  AnnotationInput,
  ExampleDetail,
  ExampleSummary,
  FailureGroup,
  PreprocessingApi,
  Tag,
  Verdict,
} from "./api";
import { errorMessage, formatNumber, humanize } from "./format";
import { ExampleDetail as ExampleDetailView } from "./example-detail";

const PAGE_SIZE = 30;

type LoadState<T> =
  | { status: "loading" }
  | { error: string; status: "error" }
  | { data: T; status: "success" };

type SaveState = "idle" | "dirty" | "saving" | "saved" | "error";

interface AnnotationDraft {
  note: string;
  tagIds: Set<string>;
  verdict: Verdict | null;
}

type SaveOperation =
  | { identity: AnnotationIdentity; input: AnnotationInput; kind: "put"; revision: number }
  | { identity: AnnotationIdentity; kind: "delete"; revision: number };

const NOTE_AUTOSAVE_DELAY_MS = 300;

export type RegisterBeforeLeave = (
  handler: () => Promise<boolean>,
) => () => void;

function groupName(group: FailureGroup): string {
  return group.label || `${humanize(group.failure_code)} · ${humanize(group.failed_step)}`;
}

function groupKey(group: FailureGroup): string {
  return JSON.stringify([group.failure_code, group.failed_step, group.cause]);
}

function AnnotationEditor({
  api,
  example,
  onExampleChange,
  onTagCreated,
  registerBeforeLeave,
  tags,
}: {
  api: PreprocessingApi;
  example: ExampleDetail;
  onExampleChange: (example: ExampleDetail) => void;
  onTagCreated: (tag: Tag) => void;
  registerBeforeLeave: RegisterBeforeLeave;
  tags: Tag[];
}) {
  const initialDraft = (): AnnotationDraft => ({
    note: example.annotation?.note ?? "",
    tagIds: new Set(example.annotation?.tags.map(({ tag_id }) => tag_id) ?? []),
    verdict: example.annotation?.verdict ?? null,
  });
  const [draft, setDraft] = useState<AnnotationDraft>(initialDraft);
  const [newTag, setNewTag] = useState("");
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [saveError, setSaveError] = useState("");
  const annotationIdentity: AnnotationIdentity = {
    corpus_sha256: example.corpus_sha256,
    decoder_output_sha256: example.decoder_output_sha256 ?? "",
    sample_id: example.sample_id,
  };
  const activeRef = useRef(true);
  const debounceRef = useRef<number | undefined>(undefined);
  const draftRef = useRef(draft);
  const exampleRef = useRef(example);
  const inFlightRef = useRef(false);
  const onExampleChangeRef = useRef(onExampleChange);
  const queuedRef = useRef<SaveOperation | null>(null);
  const revisionRef = useRef(0);
  const runQueueRef = useRef<() => void>(() => undefined);
  const saveStateRef = useRef(saveState);
  const leaveWaitersRef = useRef<Array<(canLeave: boolean) => void>>([]);
  draftRef.current = draft;
  exampleRef.current = example;
  onExampleChangeRef.current = onExampleChange;
  saveStateRef.current = saveState;

  function settleLeaveWaiters(canLeave: boolean) {
    const waiters = leaveWaitersRef.current;
    leaveWaitersRef.current = [];
    for (const resolve of waiters) resolve(canLeave);
  }

  function operationForDraft(value: AnnotationDraft, revision = revisionRef.current): SaveOperation | null {
    if (value.verdict === null) return null;
    return {
      identity: annotationIdentity,
      input: {
        note: value.note,
        tag_ids: [...value.tagIds].sort(),
        verdict: value.verdict,
      },
      kind: "put",
      revision,
    };
  }

  function queue(operation: SaveOperation) {
    queuedRef.current = operation;
    if (activeRef.current) {
      setSaveError("");
      setSaveState("saving");
    }
    runQueueRef.current();
  }

  runQueueRef.current = () => {
    if (inFlightRef.current || queuedRef.current === null) return;
    const operation = queuedRef.current;
    queuedRef.current = null;
    inFlightRef.current = true;
    const request = operation.kind === "put"
      ? api.putAnnotation(operation.identity, operation.input)
      : api.deleteAnnotation(operation.identity).then(() => null);
    void request.then(
      (annotation) => {
        inFlightRef.current = false;
        if (queuedRef.current !== null) {
          runQueueRef.current();
          return;
        }
        if (!activeRef.current || operation.revision !== revisionRef.current) return;
        setSaveState("saved");
        onExampleChangeRef.current({ ...exampleRef.current, annotation });
        settleLeaveWaiters(true);
      },
      (error: unknown) => {
        inFlightRef.current = false;
        if (queuedRef.current !== null) {
          runQueueRef.current();
          return;
        }
        if (!activeRef.current || operation.revision !== revisionRef.current) return;
        setSaveState("error");
        setSaveError(errorMessage(error));
        settleLeaveWaiters(false);
      },
    );
  };

  function updateDraft(nextDraft: AnnotationDraft, saveImmediately: boolean) {
    revisionRef.current += 1;
    draftRef.current = nextDraft;
    setDraft(nextDraft);
    setSaveError("");
    if (debounceRef.current !== undefined) {
      window.clearTimeout(debounceRef.current);
      debounceRef.current = undefined;
    }
    const operation = operationForDraft(nextDraft);
    if (operation === null) return;
    if (saveImmediately) {
      queue(operation);
      return;
    }
    setSaveState("dirty");
    debounceRef.current = window.setTimeout(() => {
      debounceRef.current = undefined;
      const latest = operationForDraft(draftRef.current);
      if (latest !== null) queue(latest);
    }, NOTE_AUTOSAVE_DELAY_MS);
  }

  function flushDraft() {
    if (debounceRef.current === undefined) return;
    window.clearTimeout(debounceRef.current);
    debounceRef.current = undefined;
    const operation = operationForDraft(draftRef.current);
    if (operation !== null) queue(operation);
  }

  function flushBeforeLeave(): Promise<boolean> {
    flushDraft();
    if (!inFlightRef.current && queuedRef.current === null) {
      return Promise.resolve(saveStateRef.current !== "error");
    }
    return new Promise((resolve) => { leaveWaitersRef.current.push(resolve); });
  }

  useEffect(() => {
    activeRef.current = true;
    return () => {
      activeRef.current = false;
      flushDraft();
    };
  }, []);

  useEffect(
    () => registerBeforeLeave(flushBeforeLeave),
    [registerBeforeLeave],
  );

  useEffect(() => {
    function warnBeforeUnload(event: BeforeUnloadEvent) {
      if (!["dirty", "saving", "error"].includes(saveStateRef.current)) return;
      event.preventDefault();
    }
    window.addEventListener("beforeunload", warnBeforeUnload);
    return () => window.removeEventListener("beforeunload", warnBeforeUnload);
  }, []);

  function chooseVerdict(verdict: Verdict) {
    updateDraft({ ...draftRef.current, verdict }, true);
  }

  function toggleTag(tagId: string, checked: boolean) {
    const nextTagIds = new Set(draftRef.current.tagIds);
    if (checked) nextTagIds.add(tagId);
    else nextTagIds.delete(tagId);
    updateDraft({ ...draftRef.current, tagIds: nextTagIds }, true);
  }

  function clearAnnotation() {
    if (debounceRef.current !== undefined) {
      window.clearTimeout(debounceRef.current);
      debounceRef.current = undefined;
    }
    revisionRef.current += 1;
    const nextDraft: AnnotationDraft = { note: "", tagIds: new Set(), verdict: null };
    draftRef.current = nextDraft;
    setDraft(nextDraft);
    queue({ identity: annotationIdentity, kind: "delete", revision: revisionRef.current });
  }

  async function createTag() {
    const name = newTag.trim();
    if (name === "") return;
    setSaveState("saving");
    setSaveError("");
    try {
      const tag = await api.createTag(name);
      onTagCreated(tag);
      setNewTag("");
      const nextTagIds = new Set(draftRef.current.tagIds).add(tag.tag_id);
      if (draftRef.current.verdict !== null) {
        updateDraft({ ...draftRef.current, tagIds: nextTagIds }, true);
      } else {
        const nextDraft = { ...draftRef.current, tagIds: nextTagIds };
        draftRef.current = nextDraft;
        setDraft(nextDraft);
        setSaveState("idle");
      }
    } catch (error) {
      setSaveState("error");
      setSaveError(errorMessage(error));
    }
  }

  const savedAnnotation: Annotation | null = example.annotation;

  return (
    <form className="annotation-editor" onSubmit={(event) => event.preventDefault()}>
      <div className="annotation-heading">
        <div>
          <p className="eyebrow">Human review</p>
          <h3>Annotation</h3>
        </div>
        <div className={`save-state save-state--${saveState}`} aria-live="polite">
          {saveState === "dirty" && "Unsaved changes"}
          {saveState === "saving" && "Saving…"}
          {saveState === "saved" && "Saved"}
          {saveState === "error" && "Save failed"}
        </div>
      </div>

      <fieldset>
        <legend>Verdict</legend>
        <label aria-disabled={example.decoder_output_sha256 === null}>
          <input
            checked={draft.verdict === "should_be_parseable"}
            disabled={example.decoder_output_sha256 === null}
            name="verdict"
            onChange={() => chooseVerdict("should_be_parseable")}
            type="radio"
          />
          <span><strong>Should be parseable</strong><small>This response contains code preprocessing should recover.</small></span>
        </label>
        <label aria-disabled={example.decoder_output_sha256 === null}>
          <input
            checked={draft.verdict === "expected_no_code"}
            disabled={example.decoder_output_sha256 === null}
            name="verdict"
            onChange={() => chooseVerdict("expected_no_code")}
            type="radio"
          />
          <span><strong>Expected no code</strong><small>The response correctly has no usable function.</small></span>
        </label>
      </fieldset>

      <label className="field-label" htmlFor="annotation-note">Note</label>
      <textarea
        disabled={draft.verdict === null || example.decoder_output_sha256 === null}
        id="annotation-note"
        onBlur={flushDraft}
        onChange={(event) => updateDraft({ ...draftRef.current, note: event.target.value }, false)}
        placeholder={draft.verdict === null ? "Choose a verdict before adding a note" : "Optional review context"}
        rows={4}
        value={draft.note}
      />

      <fieldset className="tag-fieldset" disabled={draft.verdict === null || example.decoder_output_sha256 === null}>
        <legend>Tags</legend>
        {tags.length === 0 ? <p className="empty-state">No tags yet.</p> : (
          <div className="tag-options">
            {tags.map((tag) => (
              <label key={tag.tag_id}>
                <input
                  checked={draft.tagIds.has(tag.tag_id)}
                  onChange={(event) => toggleTag(tag.tag_id, event.target.checked)}
                  type="checkbox"
                />
                {tag.name}
              </label>
            ))}
          </div>
        )}
      </fieldset>

      <div className="create-tag">
        <label htmlFor="new-tag">Create tag</label>
        <div>
          <input
            id="new-tag"
            onChange={(event) => setNewTag(event.target.value)}
            placeholder="e.g. markdown fence"
            value={newTag}
          />
          <button disabled={newTag.trim() === ""} onClick={() => void createTag()} type="button">Create and select</button>
        </div>
      </div>

      {saveError !== "" && <p className="inline-error" role="alert">{saveError}</p>}
      <button className="clear-button" disabled={example.decoder_output_sha256 === null || (savedAnnotation === null && draft.verdict === null)} onClick={clearAnnotation} type="button">
        Clear annotation
      </button>
    </form>
  );
}

export function Review({
  api,
  runId,
  tags,
  onTagCreated,
  registerBeforeLeave: outerRegisterBeforeLeave,
}: {
  api: PreprocessingApi;
  runId: string;
  tags: Tag[];
  onTagCreated: (tag: Tag) => void;
  registerBeforeLeave?: RegisterBeforeLeave;
}) {
  const [failureState, setFailureState] = useState<LoadState<FailureGroup[]>>({ status: "loading" });
  const [selectedFailureKey, setSelectedFailureKey] = useState("");
  const [exampleState, setExampleState] = useState<LoadState<ExampleSummary[]>>({ status: "loading" });
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState("");
  const [detailState, setDetailState] = useState<LoadState<ExampleDetail> | null>(null);
  const [retry, setRetry] = useState(0);
  const beforeLeaveRef = useRef<() => Promise<boolean>>(async () => true);
  const internalRegisterBeforeLeave = useCallback<RegisterBeforeLeave>((handler) => {
    beforeLeaveRef.current = handler;
    return () => {
      if (beforeLeaveRef.current === handler) {
        beforeLeaveRef.current = async () => true;
      }
    };
  }, []);
  const registerBeforeLeave = useCallback<RegisterBeforeLeave>((handler) => {
    const unregisterInternal = internalRegisterBeforeLeave(handler);
    const unregisterOuter = outerRegisterBeforeLeave?.(handler);
    return () => {
      unregisterInternal();
      unregisterOuter?.();
    };
  }, [internalRegisterBeforeLeave, outerRegisterBeforeLeave]);

  async function navigate(action: () => void) {
    if (await beforeLeaveRef.current()) action();
  }

  useEffect(() => {
    let active = true;
    setFailureState({ status: "loading" });
    setSelectedFailureKey("");
    setExampleState({ status: "loading" });
    setTotal(0);
    setOffset(0);
    setSearch("");
    setSelectedId("");
    setDetailState(null);
    void api.getFailures(runId).then(
      (response) => {
        if (!active) return;
        setFailureState({ data: response.groups, status: "success" });
        setSelectedFailureKey(response.groups[0] === undefined ? "" : groupKey(response.groups[0]));
      },
      (error: unknown) => {
        if (active) setFailureState({ error: errorMessage(error), status: "error" });
      },
    );
    return () => { active = false; };
  }, [api, retry, runId]);

  useEffect(() => {
    const group = failureState.status === "success"
      ? failureState.data.find((candidate) => groupKey(candidate) === selectedFailureKey)
      : undefined;
    if (group === undefined) {
      setExampleState({ data: [], status: "success" });
      setTotal(0);
      return;
    }
    let active = true;
    setExampleState({ status: "loading" });
    void api.getExamples(runId, {
      ...(group.cause === null ? { cause_is_null: true } : { cause: group.cause }),
      failed_step: group.failed_step,
      failure_code: group.failure_code,
      limit: PAGE_SIZE,
      offset,
      search,
    }).then(
      (response) => {
        if (!active) return;
        setExampleState({ data: response.items, status: "success" });
        setTotal(response.total);
        setSelectedId((current) => response.items.some(({ sample_id }) => sample_id === current)
          ? current
          : response.items[0]?.sample_id ?? "");
      },
      (error: unknown) => {
        if (active) setExampleState({ error: errorMessage(error), status: "error" });
      },
    );
    return () => { active = false; };
  }, [api, failureState, offset, retry, runId, search, selectedFailureKey]);

  useEffect(() => {
    if (selectedId === "") {
      setDetailState(null);
      return;
    }
    let active = true;
    setDetailState({ status: "loading" });
    void api.getExample(runId, selectedId).then(
      (example) => {
        if (active) setDetailState({ data: example, status: "success" });
      },
      (error: unknown) => {
        if (active) setDetailState({ error: errorMessage(error), status: "error" });
      },
    );
    return () => { active = false; };
  }, [api, retry, runId, selectedId]);

  const groups = failureState.status === "success" ? failureState.data : [];
  const examples = exampleState.status === "success" ? exampleState.data : [];
  const selectedIndex = examples.findIndex(({ sample_id }) => sample_id === selectedId);
  const selectedGroup = useMemo(
    () => groups.find((group) => groupKey(group) === selectedFailureKey),
    [groups, selectedFailureKey],
  );

  return (
    <section className="surface" aria-labelledby="review-title">
      <div className="surface-heading">
        <div><p className="eyebrow">Review</p><h2 id="review-title">Triage terminal preprocessing failures</h2></div>
        <a className="export-link" href="/api/annotations/export">Export annotations</a>
      </div>
      <p className="surface-copy">Review nonblank decoder outputs that produced no final function candidate. Changes save to the local annotation database.</p>

      {failureState.status === "loading" && <p className="loading-state" role="status">Loading failure groups…</p>}
      {failureState.status === "error" && (
        <div className="error-state" role="alert"><strong>Could not load failure groups.</strong><span>{failureState.error}</span><button onClick={() => setRetry((value) => value + 1)} type="button">Retry</button></div>
      )}
      {failureState.status === "success" && groups.length === 0 && <p className="empty-state">This run has no reviewable failures.</p>}

      {groups.length > 0 && (
        <div className="failure-groups" aria-label="Terminal failure groups">
          {groups.map((group) => (
            <button
              aria-pressed={groupKey(group) === selectedFailureKey}
              className="failure-group"
              key={groupKey(group)}
              onClick={() => void navigate(() => {
                  setSelectedFailureKey(groupKey(group));
                  setOffset(0);
                  setSearch("");
                  setSelectedId("");
                })}
              type="button"
            >
              <span>{groupName(group)}</span><strong>{formatNumber(group.count)}</strong><small>{humanize(group.failed_step)}</small>
            </button>
          ))}
        </div>
      )}

      {selectedGroup !== undefined && (
        <div className="review-workspace">
          <aside className="example-browser" aria-label="Failure examples">
            <label htmlFor="failure-search">Search this group</label>
            <input
              id="failure-search"
              onChange={(event) => { setSearch(event.target.value); setOffset(0); }}
              placeholder="Sample, task, output…"
              type="search"
              value={search}
            />
            <p className="result-count">{formatNumber(total)} examples</p>
            {exampleState.status === "loading" && <p className="loading-state" role="status">Loading examples…</p>}
            {exampleState.status === "error" && <p className="inline-error" role="alert">{exampleState.error}</p>}
            {exampleState.status === "success" && examples.length === 0 && <p className="empty-state">No examples match this filter.</p>}
            <div className="example-list">
              {examples.map((example) => (
                <button
                  aria-current={example.sample_id === selectedId ? "true" : undefined}
                  key={example.sample_id}
                  onClick={() => void navigate(() => setSelectedId(example.sample_id))}
                  type="button"
                >
                  <div className="example-card-heading">
                    <strong>{example.sample_id}</strong>
                    <StatusBadge status={example.annotation_verdict === null ? "neutral" : "success"}>
                      {example.annotation_verdict === null ? "unreviewed" : humanize(example.annotation_verdict)}
                    </StatusBadge>
                  </div>
                  <span>{humanize(example.outcome)}</span>
                  <small>{example.raw_preview || "No output preview"}</small>
                </button>
              ))}
            </div>
            <div className="pagination" aria-label="Example pagination">
              <button disabled={offset === 0} onClick={() => void navigate(() => setOffset(Math.max(0, offset - PAGE_SIZE)))} type="button">Previous</button>
              <span>{total === 0 ? "0–0" : `${offset + 1}–${Math.min(offset + PAGE_SIZE, total)}`} of {total}</span>
              <button disabled={offset + PAGE_SIZE >= total} onClick={() => void navigate(() => setOffset(offset + PAGE_SIZE))} type="button">Next</button>
            </div>
          </aside>

          <div className="review-detail">
            <div className="sequential-nav" aria-label="Sequential review navigation">
              <button disabled={selectedIndex <= 0} onClick={() => void navigate(() => setSelectedId(examples[selectedIndex - 1]?.sample_id ?? selectedId))} type="button">← Previous example</button>
              <button disabled={selectedIndex < 0 || selectedIndex >= examples.length - 1} onClick={() => void navigate(() => setSelectedId(examples[selectedIndex + 1]?.sample_id ?? selectedId))} type="button">Next example →</button>
            </div>
            {detailState?.status === "loading" && <p className="loading-state" role="status">Loading example detail…</p>}
            {detailState?.status === "error" && <p className="inline-error" role="alert">{detailState.error}</p>}
            {detailState?.status === "success" && (
              <>
                <AnnotationEditor
                  api={api}
                  example={detailState.data}
                  key={`${detailState.data.corpus_sha256}:${detailState.data.sample_id}:${detailState.data.decoder_output_sha256 ?? "missing"}`}
                  onExampleChange={(example) => {
                    setDetailState({ data: example, status: "success" });
                    setExampleState((current) => current.status !== "success" ? current : {
                      data: current.data.map((summary) => summary.sample_id === example.sample_id
                        ? { ...summary, annotation_verdict: example.annotation?.verdict ?? null }
                        : summary),
                      status: "success",
                    });
                  }}
                  onTagCreated={onTagCreated}
                  registerBeforeLeave={registerBeforeLeave}
                  tags={tags}
                />
                <ExampleDetailView example={detailState.data} />
              </>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
