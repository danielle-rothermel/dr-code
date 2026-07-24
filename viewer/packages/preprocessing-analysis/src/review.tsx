import { CodeBlock, StatusBadge } from "@dr-code/viewer";
import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";

import type {
  Annotation,
  AnnotationIdentity,
  AnnotationInput,
  ExampleDetail,
  FailureGroup,
  PreprocessingApi,
  Tag,
  Verdict,
} from "./api";
import {
  ANNOTATION_NOTE_MAX_LENGTH,
  ANNOTATION_TAG_IDS_MAX_COUNT,
  isAnnotationNoteInContract,
  isTagNameInContract,
  normalizeTagName,
} from "./annotation-contract";
import { CandidateOrigins } from "./candidate-origins";
import { errorMessage, formatNumber, humanize } from "./format";
import { TaskAnnotationEditor } from "./task-annotation";
import { useAutosaveQueue } from "./use-autosave-queue";

const DEFAULT_PAGE_SIZE = 10;
const PAGE_SIZES = [10, 25, 50] as const;
type LoadState<T> =
  | { status: "loading" }
  | { error: string; status: "error" }
  | { data: T; status: "success" };

interface AnnotationDraft {
  note: string;
  tagIds: Set<string>;
  verdict: Verdict | null;
}

type AnnotationMutation =
  | { input: AnnotationInput; kind: "put" }
  | { kind: "delete" };

export interface CardGuard {
  flush: () => Promise<boolean>;
  isUnsafe: () => boolean;
}

export type RegisterCardGuard = (
  key: string,
  guard: CardGuard,
) => () => void;

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
          <CandidateOrigins origins={candidate.origins} />
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
    ? humanize(example.failure_code ?? "failure")
    : example.cause;

  return (
    <div className="review-example-main">
      <div className="failure-reason">
        <StatusBadge className="failure-reason__badge" status="failure">
          {failureReason} · {humanize(example.failed_step ?? "unknown_step")}
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
  const [saveError, setSaveError] = useState("");
  const [tagSaveState, setTagSaveState] = useState<"idle" | "saving" | "error">("idle");
  const [tagSaveError, setTagSaveError] = useState("");
  const annotationIdentity: AnnotationIdentity | null = example.decoder_output_sha256 === null ? null : {
    corpus_sha256: example.corpus_sha256,
    decoder_output_sha256: example.decoder_output_sha256,
    sample_id: example.sample_id,
  };
  const activeRef = useRef(true);
  const draftRef = useRef(draft);
  const exampleRef = useRef(example);
  const onExampleChangeRef = useRef(onExampleChange);
  const tagCreationRef = useRef<Promise<boolean> | null>(null);
  const tagSaveErrorRef = useRef("");
  const pendingTagNameRef = useRef<string | null>(null);
  draftRef.current = draft;
  exampleRef.current = example;
  onExampleChangeRef.current = onExampleChange;

  function updateTagSaveError(message: string) {
    tagSaveErrorRef.current = message;
    if (activeRef.current) setTagSaveError(message);
  }

  function inputForDraft(value: AnnotationDraft): AnnotationInput {
    return {
      note: value.note,
      tag_ids: [...value.tagIds].sort(),
      verdict: value.verdict,
    };
  }

  const identityKey = annotationIdentity === null
    ? `${example.corpus_sha256}:${example.sample_id}:missing`
    : `${annotationIdentity.corpus_sha256}:${annotationIdentity.sample_id}:${annotationIdentity.decoder_output_sha256}`;
  const autosave = useAutosaveQueue<AnnotationMutation, Annotation | null>({
    onError: (error) => {
      if (activeRef.current) setSaveError(errorMessage(error));
    },
    onSaved: (annotation) => {
      if (!activeRef.current) return;
      if (annotation === null) {
        const emptyDraft: AnnotationDraft = {
          note: "",
          tagIds: new Set(),
          verdict: null,
        };
        draftRef.current = emptyDraft;
        setDraft(emptyDraft);
      }
      onExampleChangeRef.current({ ...exampleRef.current, annotation });
    },
    save: async (mutation) => {
      if (annotationIdentity === null) {
        throw new Error("annotation output is unavailable");
      }
      if (mutation.kind === "delete") {
        await api.deleteAnnotation(annotationIdentity);
        return null;
      }
      return api.putAnnotation(annotationIdentity, mutation.input);
    },
    scopeKey: identityKey,
  });

  function updateDraft(nextDraft: AnnotationDraft, saveImmediately: boolean) {
    if (
      !isAnnotationNoteInContract(nextDraft.note)
      || nextDraft.tagIds.size > ANNOTATION_TAG_IDS_MAX_COUNT
    ) return;
    draftRef.current = nextDraft;
    setDraft(nextDraft);
    setSaveError("");
    if (annotationIdentity !== null) {
      autosave.edit(
        { input: inputForDraft(nextDraft), kind: "put" },
        saveImmediately,
      );
    }
  }

  function flushDraft() {
    void autosave.flush();
  }

  async function flushBeforeLeave(): Promise<boolean> {
    if (tagCreationRef.current !== null) {
      if (!await tagCreationRef.current) return false;
    } else if (tagSaveErrorRef.current !== "") {
      const name = pendingTagNameRef.current;
      if (name === null || !await startTagCreation(name)) return false;
    }
    return autosave.flush();
  }

  useEffect(() => {
    activeRef.current = true;
    return () => {
      activeRef.current = false;
    };
  }, []);

  useEffect(
    () => registerCardGuard(
      `${example.corpus_sha256}:${example.sample_id}:${example.decoder_output_sha256 ?? "missing"}`,
      {
        flush: flushBeforeLeave,
        isUnsafe: () => (
          tagCreationRef.current !== null
          || tagSaveErrorRef.current !== ""
          || autosave.isUnsafe()
        ),
      },
    ),
    [example.corpus_sha256, example.decoder_output_sha256, example.sample_id, registerCardGuard],
  );

  function chooseVerdict(verdict: Verdict | null) {
    updateDraft({ ...draftRef.current, verdict }, true);
  }

  function toggleTag(tagId: string, checked: boolean) {
    const nextTagIds = new Set(draftRef.current.tagIds);
    if (
      checked
      && !nextTagIds.has(tagId)
      && nextTagIds.size >= ANNOTATION_TAG_IDS_MAX_COUNT
    ) return;
    if (checked) nextTagIds.add(tagId);
    else nextTagIds.delete(tagId);
    updateDraft({ ...draftRef.current, tagIds: nextTagIds }, true);
  }

  function startTagCreation(name: string): Promise<boolean> {
    if (annotationIdentity === null) return Promise.resolve(false);
    if (tagCreationRef.current !== null) return tagCreationRef.current;
    const normalizedName = normalizeTagName(name);
    if (
      !isTagNameInContract(normalizedName)
      || draftRef.current.tagIds.size >= ANNOTATION_TAG_IDS_MAX_COUNT
    ) return Promise.resolve(false);
    pendingTagNameRef.current = normalizedName;
    updateTagSaveError("");
    if (activeRef.current) setTagSaveState("saving");
    const request = Promise.resolve().then(() => api.createTag(normalizedName)).then(
      (tag) => {
        if (!activeRef.current) return false;
        onTagCreated(tag);
        pendingTagNameRef.current = null;
        setNewTag("");
        setTagSaveState("idle");
        const nextTagIds = new Set(draftRef.current.tagIds);
        if (
          nextTagIds.has(tag.tag_id)
          || nextTagIds.size < ANNOTATION_TAG_IDS_MAX_COUNT
        ) {
          nextTagIds.add(tag.tag_id);
          updateDraft({ ...draftRef.current, tagIds: nextTagIds }, true);
        }
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
    if (!isTagNameInContract(newTag) || annotationIdentity === null) return;
    void startTagCreation(newTag);
  }

  function discardTagCreationIntent() {
    pendingTagNameRef.current = null;
    updateTagSaveError("");
    if (activeRef.current) {
      setNewTag("");
      setTagSaveState("idle");
    }
  }

  function deleteSavedAnnotation() {
    if (
      annotationIdentity === null
      || tagCreationRef.current !== null
    ) return;
    discardTagCreationIntent();
    setSaveError("");
    autosave.edit({ kind: "delete" }, true);
  }

  const disabled = annotationIdentity === null;
  const displayedSaveState = tagSaveState === "error"
    ? "error"
    : tagSaveState === "saving" ? "saving" : autosave.saveState;

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
        maxLength={ANNOTATION_NOTE_MAX_LENGTH * 2}
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
          <button
            disabled={
              disabled
              || !isTagNameInContract(newTag)
              || draft.tagIds.size >= ANNOTATION_TAG_IDS_MAX_COUNT
              || tagSaveState === "saving"
            }
            onClick={createTag}
            type="button"
          >
            {tagSaveState === "error" ? "Retry create and select" : "Create and select"}
          </button>
        </div>
      </div>

      <fieldset
        className="tag-fieldset"
        disabled={disabled || tagSaveState === "saving"}
      >
        <legend>Tags</legend>
        {tags.length === 0 ? <p className="empty-state">No tags yet.</p> : (
          <div className="tag-options">
            {tags.map((tag) => (
              <label key={tag.tag_id}>
                <input
                  checked={draft.tagIds.has(tag.tag_id)}
                  disabled={
                    !draft.tagIds.has(tag.tag_id)
                    && draft.tagIds.size >= ANNOTATION_TAG_IDS_MAX_COUNT
                  }
                  onChange={(event) => toggleTag(tag.tag_id, event.target.checked)}
                  type="checkbox"
                />
                {tag.name}
              </label>
            ))}
          </div>
        )}
      </fieldset>

      <button
        disabled={
          disabled
          || example.annotation === null
          || autosave.currentSaveState() === "saving"
        }
        onClick={deleteSavedAnnotation}
        type="button"
      >
        Delete annotation
      </button>

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
  const [navigationPending, setNavigationPending] = useState(false);
  const cardGuardsRef = useRef(new Map<string, CardGuard>());
  const navigationInFlightRef = useRef(false);

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
    if (navigationInFlightRef.current) return;
    navigationInFlightRef.current = true;
    setNavigationPending(true);
    try {
      if (await flushAllCards()) action();
    } finally {
      navigationInFlightRef.current = false;
      setNavigationPending(false);
    }
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
  const taskIdentities = useMemo(() => {
    const identities = new Map<
      string,
      { dataset_id: string; task_id: string; task_identity: string }
    >();
    for (const example of examples) {
      const taskId = example.context.task_id;
      if (
        example.dataset_id === null
        || example.task_identity === null
        || typeof taskId !== "string"
      ) continue;
      const identity = {
        dataset_id: example.dataset_id,
        task_id: taskId,
        task_identity: example.task_identity,
      };
      identities.set(
        JSON.stringify([
          identity.dataset_id,
          identity.task_id,
          identity.task_identity,
        ]),
        identity,
      );
    }
    return [...identities.values()];
  }, [examples]);
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
          <div
            aria-busy={navigationPending}
            aria-label="Review controls"
            className="review-toolbar"
          >
            <label>
              <span>Failure group</span>
              <select
                aria-label="Failure group"
                disabled={navigationPending}
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
                  disabled={navigationPending}
                  id="review-search"
                  onChange={(event) => setSearchDraft(event.target.value)}
                  placeholder="Sample, task, output…"
                  type="search"
                  value={searchDraft}
                />
                <button disabled={navigationPending} type="submit">Search</button>
              </div>
            </form>
            <label>
              <span>Page</span>
              <select
                aria-label="Page number"
                disabled={navigationPending}
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
                disabled={navigationPending}
                onChange={(event) => {
                  const nextPageSize = Number(event.target.value);
                  void navigate(() => { setPageSize(nextPageSize); setPage(1); });
                }}
                value={pageSize}
              >
                {PAGE_SIZES.map((size) => <option key={size} value={size}>{size}</option>)}
              </select>
            </label>
            <button disabled={navigationPending || page <= 1} onClick={() => void navigate(() => setPage((value) => value - 1))} type="button">Previous page</button>
            <button disabled={navigationPending || page >= pageCount} onClick={() => void navigate(() => setPage((value) => value + 1))} type="button">Next page</button>
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
            <>
              <section
                aria-label="Task annotations for this page"
                className="page-task-annotations"
              >
                <h3>Task annotations</h3>
                <div className="page-task-annotation-grid">
                  {taskIdentities.map((identity) => (
                    <TaskAnnotationEditor
                      api={api}
                      identity={identity}
                      key={JSON.stringify([
                        identity.dataset_id,
                        identity.task_id,
                        identity.task_identity,
                      ])}
                      onTagCreated={onTagCreated}
                      registerCardGuard={registerCardGuard}
                      tags={tags}
                    />
                  ))}
                </div>
              </section>
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
            </>
          )}
        </>
      )}
    </section>
  );
}
