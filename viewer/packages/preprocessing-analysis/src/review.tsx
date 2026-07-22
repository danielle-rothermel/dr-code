import { CodeBlock, StatusBadge } from "@dr-code/viewer";
import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";

import type {
  AnnotationIdentity,
  AnnotationInput,
  ExampleDetail,
  FailureGroup,
  PreprocessingApi,
  Tag,
  Verdict,
} from "./api";
import { errorMessage, formatNumber, humanize } from "./format";

const DEFAULT_PAGE_SIZE = 10;
const PAGE_SIZES = [10, 25, 50] as const;
const NOTE_AUTOSAVE_DELAY_MS = 300;

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

interface SaveOperation {
  identity: AnnotationIdentity;
  input: AnnotationInput;
  revision: number;
}

interface CardGuard {
  flush: () => Promise<boolean>;
  isUnsafe: () => boolean;
}

type RegisterCardGuard = (key: string, guard: CardGuard) => () => void;

export type RegisterBeforeLeave = (
  handler: () => Promise<boolean>,
) => () => void;

function groupName(group: FailureGroup): string {
  return group.label || `${humanize(group.failure_code)} · ${humanize(group.failed_step)}`;
}

function groupKey(group: FailureGroup): string {
  return JSON.stringify([group.failure_code, group.failed_step, group.cause]);
}

function metadataFieldClass(key: string, value: string | number | boolean | null): string {
  if (/warnings?/i.test(key)) return "metadata-field metadata-field--full";
  if (key === "source_record_id" || key === "content_sha256") {
    return "metadata-field metadata-field--half";
  }
  if (typeof value === "boolean" || /(?:kind|type|status|language|mode|split)$/i.test(key)) {
    return "metadata-field metadata-field--compact";
  }
  return "metadata-field";
}

function Diagnostics({ example }: { example: ExampleDetail }) {
  if (example.facts.length === 0 && example.rejections.length === 0) return null;

  return (
    <section className="review-diagnostics" aria-label={`Diagnostics for ${example.sample_id}`}>
      <h4>Diagnostics</h4>
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
              <StatusBadge status="neutral">fact</StatusBadge>
              <strong>{humanize(fact.step_name)}</strong>
            </div>
            <CodeBlock code={fact.facts_json} lang="json" />
          </article>
        ))}
      </div>
    </section>
  );
}

function Candidates({ example }: { example: ExampleDetail }) {
  return (
    <section className="review-candidates" aria-label={`Candidates for ${example.sample_id}`}>
      <h4>Final candidates</h4>
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
  );
}

function ReviewExampleMain({ example }: { example: ExampleDetail }) {
  const failureReason = example.cause === null || example.cause === ""
    ? humanize(example.failure_code)
    : example.cause;

  return (
    <div className="review-example-main">
      <div className="failure-reason">
        <StatusBadge className="failure-reason__badge" status="failure">
          {failureReason} · {humanize(example.failed_step)}
        </StatusBadge>
      </div>

      <section className="decoder-output" aria-label={`Decoder output for ${example.sample_id}`}>
        <h4>Decoder output</h4>
        {example.raw_decoder_output === null ? (
          <p className="empty-state">No decoder output was present.</p>
        ) : (
          <CodeBlock code={example.raw_decoder_output} lang="python" />
        )}
      </section>

      <details className="review-details">
        <summary>Example details</summary>
        <dl className="review-metadata-grid">
          <div className="metadata-field metadata-field--full">
            <dt>Sample ID</dt>
            <dd>{example.sample_id}</dd>
          </div>
          {Object.entries(example.context).map(([key, value]) => (
            <div className={metadataFieldClass(key, value)} key={key}>
              <dt>{humanize(key)}</dt>
              <dd>{value === null ? "not recorded" : String(value)}</dd>
            </div>
          ))}
        </dl>

        <Diagnostics example={example} />
        <Candidates example={example} />
      </details>
    </div>
  );
}

function AnnotationEditor({
  api,
  example,
  onExampleChange,
  onTagCreated,
  registerCardGuard,
  tags,
}: {
  api: PreprocessingApi;
  example: ExampleDetail;
  onExampleChange: (example: ExampleDetail) => void;
  onTagCreated: (tag: Tag) => void;
  registerCardGuard: RegisterCardGuard;
  tags: Tag[];
}) {
  const controlId = useId();
  const initialDraft = (): AnnotationDraft => ({
    note: example.annotation?.note ?? "",
    tagIds: new Set(example.annotation?.tags.map(({ tag_id }) => tag_id) ?? []),
    verdict: example.annotation?.verdict ?? null,
  });
  const [draft, setDraft] = useState<AnnotationDraft>(initialDraft);
  const [newTag, setNewTag] = useState("");
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [saveError, setSaveError] = useState("");
  const [tagSaveState, setTagSaveState] = useState<"idle" | "saving" | "error">("idle");
  const [tagSaveError, setTagSaveError] = useState("");
  const annotationIdentity: AnnotationIdentity | null = example.decoder_output_sha256 === null ? null : {
    corpus_sha256: example.corpus_sha256,
    decoder_output_sha256: example.decoder_output_sha256,
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
  const tagCreationRef = useRef<Promise<boolean> | null>(null);
  const tagSaveErrorRef = useRef("");
  const pendingTagNameRef = useRef<string | null>(null);
  const leaveWaitersRef = useRef<Array<(canLeave: boolean) => void>>([]);
  draftRef.current = draft;
  exampleRef.current = example;
  onExampleChangeRef.current = onExampleChange;

  function updateSaveState(next: SaveState) {
    saveStateRef.current = next;
    if (activeRef.current) setSaveState(next);
  }

  function updateTagSaveError(message: string) {
    tagSaveErrorRef.current = message;
    if (activeRef.current) setTagSaveError(message);
  }

  function settleLeaveWaiters(canLeave: boolean) {
    const waiters = leaveWaitersRef.current;
    leaveWaitersRef.current = [];
    for (const resolve of waiters) resolve(canLeave);
  }

  function operationForDraft(value: AnnotationDraft, revision = revisionRef.current): SaveOperation | null {
    if (annotationIdentity === null) return null;
    return {
      identity: annotationIdentity,
      input: {
        note: value.note,
        tag_ids: [...value.tagIds].sort(),
        verdict: value.verdict,
      },
      revision,
    };
  }

  function queue(operation: SaveOperation) {
    queuedRef.current = operation;
    if (activeRef.current) setSaveError("");
    updateSaveState("saving");
    runQueueRef.current();
  }

  runQueueRef.current = () => {
    if (inFlightRef.current || queuedRef.current === null) return;
    const operation = queuedRef.current;
    queuedRef.current = null;
    inFlightRef.current = true;
    void api.putAnnotation(operation.identity, operation.input).then(
      (annotation) => {
        inFlightRef.current = false;
        if (operation.revision === revisionRef.current) {
          updateSaveState("saved");
          if (activeRef.current) {
            onExampleChangeRef.current({ ...exampleRef.current, annotation });
          }
        }
        if (queuedRef.current !== null) {
          runQueueRef.current();
          return;
        }
        if (debounceRef.current === undefined) settleLeaveWaiters(true);
      },
      (error: unknown) => {
        inFlightRef.current = false;
        if (queuedRef.current !== null) {
          runQueueRef.current();
          return;
        }
        if (operation.revision !== revisionRef.current && debounceRef.current !== undefined) return;
        updateSaveState("error");
        if (activeRef.current) setSaveError(errorMessage(error));
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
    updateSaveState("dirty");
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

  async function flushBeforeLeave(): Promise<boolean> {
    if (tagCreationRef.current !== null) {
      if (!await tagCreationRef.current) return false;
    } else if (tagSaveErrorRef.current !== "") {
      const name = pendingTagNameRef.current;
      if (name === null || !await startTagCreation(name)) return false;
    }
    flushDraft();
    if (saveStateRef.current === "error" && !inFlightRef.current && queuedRef.current === null) {
      const retry = operationForDraft(draftRef.current);
      if (retry !== null) queue(retry);
    }
    if (!inFlightRef.current && queuedRef.current === null) {
      return saveStateRef.current !== "error";
    }
    return new Promise((resolve) => { leaveWaitersRef.current.push(resolve); });
  }

  useEffect(() => {
    activeRef.current = true;
    return () => {
      flushDraft();
      activeRef.current = false;
    };
  }, []);

  useEffect(
    () => registerCardGuard(
      `${example.corpus_sha256}:${example.sample_id}:${example.decoder_output_sha256 ?? "missing"}`,
      {
        flush: flushBeforeLeave,
        isUnsafe: () => tagCreationRef.current !== null || tagSaveErrorRef.current !== "" || ["dirty", "saving", "error"].includes(saveStateRef.current),
      },
    ),
    [example.corpus_sha256, example.decoder_output_sha256, example.sample_id, registerCardGuard],
  );

  function chooseVerdict(verdict: Verdict | null) {
    updateDraft({ ...draftRef.current, verdict }, true);
  }

  function toggleTag(tagId: string, checked: boolean) {
    const nextTagIds = new Set(draftRef.current.tagIds);
    if (checked) nextTagIds.add(tagId);
    else nextTagIds.delete(tagId);
    updateDraft({ ...draftRef.current, tagIds: nextTagIds }, true);
  }

  function startTagCreation(name: string): Promise<boolean> {
    if (annotationIdentity === null) return Promise.resolve(false);
    if (tagCreationRef.current !== null) return tagCreationRef.current;
    pendingTagNameRef.current = name;
    updateTagSaveError("");
    if (activeRef.current) setTagSaveState("saving");
    const request = Promise.resolve().then(() => api.createTag(name)).then(
      (tag) => {
        if (!activeRef.current) return false;
        onTagCreated(tag);
        pendingTagNameRef.current = null;
        setNewTag("");
        setTagSaveState("idle");
        updateDraft({ ...draftRef.current, tagIds: new Set(draftRef.current.tagIds).add(tag.tag_id) }, true);
        return true;
      },
      (error: unknown) => {
        updateTagSaveError(errorMessage(error));
        if (activeRef.current) setTagSaveState("error");
        return false;
      },
    ).finally(() => { tagCreationRef.current = null; });
    tagCreationRef.current = request;
    return request;
  }

  function createTag() {
    const name = newTag.trim();
    if (name === "" || annotationIdentity === null) return;
    void startTagCreation(name);
  }

  const disabled = annotationIdentity === null;
  const displayedSaveState = tagSaveState === "error"
    ? "error"
    : tagSaveState === "saving" ? "saving" : saveState;

  return (
    <form className="annotation-editor" onSubmit={(event) => event.preventDefault()}>
      <div className="annotation-heading">
        <h3>Annotation</h3>
        <div className={`save-state save-state--${displayedSaveState}`} aria-live="polite">
          {tagSaveState === "error" && "Tag save failed"}
          {tagSaveState !== "error" && displayedSaveState === "dirty" && "Unsaved changes"}
          {tagSaveState !== "error" && displayedSaveState === "saving" && "Saving…"}
          {tagSaveState !== "error" && displayedSaveState === "saved" && "Saved"}
          {tagSaveState !== "error" && displayedSaveState === "error" && "Save failed"}
        </div>
      </div>

      <fieldset className="verdict-options" disabled={disabled}>
        <legend className="visually-hidden">Verdict</legend>
        <label>
          <input checked={draft.verdict === null} name={`${controlId}-verdict`} onChange={() => chooseVerdict(null)} type="radio" />
          Unlabeled
        </label>
        <label>
          <input checked={draft.verdict === "should_be_parseable"} name={`${controlId}-verdict`} onChange={() => chooseVerdict("should_be_parseable")} type="radio" />
          Flag
        </label>
        <label>
          <input checked={draft.verdict === "expected_no_code"} name={`${controlId}-verdict`} onChange={() => chooseVerdict("expected_no_code")} type="radio" />
          Verify
        </label>
      </fieldset>

      <label className="field-label" htmlFor={`${controlId}-note`}>Comment</label>
      <textarea
        disabled={disabled}
        id={`${controlId}-note`}
        onBlur={flushDraft}
        onChange={(event) => updateDraft({ ...draftRef.current, note: event.target.value }, false)}
        placeholder="Optional review context"
        rows={4}
        value={draft.note}
      />

      <div className="create-tag">
        <label htmlFor={`${controlId}-new-tag`}>Create tag</label>
        <div>
          <input
            disabled={disabled}
            id={`${controlId}-new-tag`}
            onChange={(event) => setNewTag(event.target.value)}
            placeholder="e.g. markdown fence"
            value={newTag}
          />
          <button disabled={disabled || newTag.trim() === "" || tagSaveState === "saving"} onClick={createTag} type="button">
            {tagSaveState === "error" ? "Retry create and select" : "Create and select"}
          </button>
        </div>
      </div>

      <fieldset className="tag-fieldset" disabled={disabled}>
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

      {saveError !== "" && <p className="inline-error" role="alert">{saveError}</p>}
      {tagSaveError !== "" && <p className="inline-error" role="alert">Tag creation failed: {tagSaveError}</p>}
    </form>
  );
}

function ReviewExampleCard({
  api,
  example,
  onExampleChange,
  onTagCreated,
  registerCardGuard,
  tags,
}: {
  api: PreprocessingApi;
  example: ExampleDetail;
  onExampleChange: (example: ExampleDetail) => void;
  onTagCreated: (tag: Tag) => void;
  registerCardGuard: RegisterCardGuard;
  tags: Tag[];
}) {
  return (
    <article className="review-example-card review-example-card--three-one" aria-label={`Example ${example.sample_id}`}>
      <ReviewExampleMain example={example} />
      <aside className="annotation-rail" aria-label={`Annotation for ${example.sample_id}`}>
        <AnnotationEditor
          api={api}
          example={example}
          onExampleChange={onExampleChange}
          onTagCreated={onTagCreated}
          registerCardGuard={registerCardGuard}
          tags={tags}
        />
      </aside>
    </article>
  );
}

export function Review({
  api,
  runId,
  tags,
  onTagCreated,
  registerBeforeLeave,
}: {
  api: PreprocessingApi;
  runId: string;
  tags: Tag[];
  onTagCreated: (tag: Tag) => void;
  registerBeforeLeave?: RegisterBeforeLeave;
}) {
  const [failureState, setFailureState] = useState<LoadState<FailureGroup[]>>({ status: "loading" });
  const [selectedFailureKey, setSelectedFailureKey] = useState("");
  const [exampleState, setExampleState] = useState<LoadState<ExampleDetail[]>>({ status: "loading" });
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [searchDraft, setSearchDraft] = useState("");
  const [appliedSearch, setAppliedSearch] = useState("");
  const [failureRetry, setFailureRetry] = useState(0);
  const [exampleRetry, setExampleRetry] = useState(0);
  const [navigationError, setNavigationError] = useState("");
  const cardGuardsRef = useRef(new Map<string, CardGuard>());

  const registerCardGuard = useCallback<RegisterCardGuard>((key, guard) => {
    cardGuardsRef.current.set(key, guard);
    return () => {
      if (cardGuardsRef.current.get(key) === guard) cardGuardsRef.current.delete(key);
    };
  }, []);

  const flushAllCards = useCallback(async (): Promise<boolean> => {
    const guards = [...cardGuardsRef.current.values()];
    const results = await Promise.all(guards.map(async ({ flush }) => {
      try {
        return await flush();
      } catch {
        return false;
      }
    }));
    const canLeave = results.every(Boolean);
    setNavigationError(canLeave ? "" : "Some annotations could not be saved. Your drafts are still here; retry after resolving the save error.");
    return canLeave;
  }, []);

  useEffect(
    () => registerBeforeLeave?.(flushAllCards),
    [flushAllCards, registerBeforeLeave],
  );

  useEffect(() => {
    function warnBeforeUnload(event: BeforeUnloadEvent) {
      if (![...cardGuardsRef.current.values()].some(({ isUnsafe }) => isUnsafe())) return;
      event.preventDefault();
    }
    window.addEventListener("beforeunload", warnBeforeUnload);
    return () => window.removeEventListener("beforeunload", warnBeforeUnload);
  }, []);

  async function navigate(action: () => void) {
    if (await flushAllCards()) action();
  }

  useEffect(() => {
    let active = true;
    setFailureState({ status: "loading" });
    setSelectedFailureKey("");
    setExampleState({ status: "loading" });
    setTotal(0);
    setPage(1);
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
  }, [api, failureRetry, runId]);

  const groups = failureState.status === "success" ? failureState.data : [];
  const selectedGroup = useMemo(
    () => groups.find((group) => groupKey(group) === selectedFailureKey),
    [groups, selectedFailureKey],
  );

  useEffect(() => {
    if (selectedGroup === undefined) {
      setExampleState({ data: [], status: "success" });
      setTotal(0);
      return;
    }
    let active = true;
    setExampleState({ status: "loading" });
    void api.getReviewExamples(runId, {
      ...(selectedGroup.cause === null ? { cause_is_null: true } : { cause: selectedGroup.cause }),
      failed_step: selectedGroup.failed_step,
      failure_code: selectedGroup.failure_code,
      limit: pageSize,
      offset: (page - 1) * pageSize,
      search: appliedSearch,
    }).then(
      (response) => {
        if (!active) return;
        const responsePageCount = Math.max(1, Math.ceil(response.total / pageSize));
        if (page > responsePageCount) {
          setPage(responsePageCount);
          return;
        }
        setExampleState({ data: response.items, status: "success" });
        setTotal(response.total);
      },
      (error: unknown) => {
        if (active) setExampleState({ error: errorMessage(error), status: "error" });
      },
    );
    return () => { active = false; };
  }, [api, appliedSearch, exampleRetry, page, pageSize, runId, selectedGroup]);

  const examples = exampleState.status === "success" ? exampleState.data : [];
  const pageCount = Math.max(1, Math.ceil(total / pageSize));

  return (
    <section className="surface review-surface" aria-labelledby="review-title">
      <div className="surface-heading review-heading">
        <div><p className="eyebrow">Review</p><h2 id="review-title">Triage terminal preprocessing failures</h2></div>
        <a className="export-link" href="/api/annotations/export">Export annotations</a>
      </div>
      <p className="surface-copy">Review nonblank decoder outputs that produced no final function candidate. Changes save to the local annotation database.</p>

      {failureState.status === "loading" && <p className="loading-state" role="status">Loading failure groups…</p>}
      {failureState.status === "error" && (
        <div className="error-state" role="alert"><strong>Could not load failure groups.</strong><span>{failureState.error}</span><button onClick={() => setFailureRetry((value) => value + 1)} type="button">Retry</button></div>
      )}
      {failureState.status === "success" && groups.length === 0 && <p className="empty-state">This run has no reviewable failures.</p>}

      {groups.length > 0 && selectedGroup !== undefined && (
        <>
          <div className="review-toolbar" aria-label="Review controls">
            <label>
              <span>Failure group</span>
              <select
                aria-label="Failure group"
                onChange={(event) => {
                  const key = event.target.value;
                  void navigate(() => { setSelectedFailureKey(key); setPage(1); });
                }}
                value={selectedFailureKey}
              >
                {groups.map((group) => (
                  <option key={groupKey(group)} value={groupKey(group)}>{groupName(group)} ({formatNumber(group.count)})</option>
                ))}
              </select>
            </label>
            <form
              aria-label="Search review examples"
              className="review-search"
              onSubmit={(event) => {
                event.preventDefault();
                const nextSearch = searchDraft;
                void navigate(() => { setAppliedSearch(nextSearch); setPage(1); });
              }}
              role="search"
            >
              <label htmlFor="review-search"><span>Search</span></label>
              <div className="review-search__controls">
                <input
                  id="review-search"
                  onChange={(event) => setSearchDraft(event.target.value)}
                  placeholder="Sample, task, output…"
                  type="search"
                  value={searchDraft}
                />
                <button type="submit">Search</button>
              </div>
            </form>
            <label>
              <span>Page</span>
              <select
                aria-label="Page number"
                onChange={(event) => {
                  const nextPage = Number(event.target.value);
                  void navigate(() => setPage(nextPage));
                }}
                value={page}
              >
                {Array.from({ length: pageCount }, (_, index) => index + 1).map((pageNumber) => (
                  <option key={pageNumber} value={pageNumber}>Page {pageNumber} of {pageCount}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Page size</span>
              <select
                aria-label="Page size"
                onChange={(event) => {
                  const nextPageSize = Number(event.target.value);
                  void navigate(() => { setPageSize(nextPageSize); setPage(1); });
                }}
                value={pageSize}
              >
                {PAGE_SIZES.map((size) => <option key={size} value={size}>{size}</option>)}
              </select>
            </label>
            <button disabled={page <= 1} onClick={() => void navigate(() => setPage((value) => value - 1))} type="button">Previous page</button>
            <button disabled={page >= pageCount} onClick={() => void navigate(() => setPage((value) => value + 1))} type="button">Next page</button>
          </div>
          <p className="review-page-summary">Page {page} of {pageCount} · {formatNumber(total)} examples</p>

          {navigationError !== "" && (
            <div className="error-state navigation-error" role="alert">
              <strong>Navigation blocked</strong>
              <span>{navigationError}</span>
              <button onClick={() => void flushAllCards()} type="button">Retry pending saves</button>
            </div>
          )}
          {exampleState.status === "loading" && <p className="loading-state" role="status">Loading examples…</p>}
          {exampleState.status === "error" && (
            <div className="error-state" role="alert"><strong>Could not load review examples.</strong><span>{exampleState.error}</span><button onClick={() => setExampleRetry((value) => value + 1)} type="button">Retry</button></div>
          )}
          {exampleState.status === "success" && examples.length === 0 && <p className="empty-state">No examples match this filter.</p>}
          {exampleState.status === "success" && examples.length > 0 && (
            <div className="review-example-stack">
              {examples.map((example) => (
                <ReviewExampleCard
                  api={api}
                  example={example}
                  key={`${example.corpus_sha256}:${example.sample_id}:${example.decoder_output_sha256 ?? "missing"}`}
                  onExampleChange={(updated) => {
                    setExampleState((current) => current.status !== "success" ? current : {
                      data: current.data.map((item) => item.sample_id === updated.sample_id ? updated : item),
                      status: "success",
                    });
                  }}
                  onTagCreated={onTagCreated}
                  registerCardGuard={registerCardGuard}
                  tags={tags}
                />
              ))}
            </div>
          )}
        </>
      )}
    </section>
  );
}
