import { useEffect, useId, useRef, useState } from "react";

import type {
  PreprocessingApi,
  Tag,
  TaskAnnotation,
  TaskAnnotationIdentity,
  TaskAnnotationInput,
} from "./api";
import { datasetIdOf } from "./api";
import { errorMessage } from "./format";

const NOTE_AUTOSAVE_DELAY_MS = 300;

type SaveState = "idle" | "dirty" | "saving" | "saved" | "error";

interface TaskDraft {
  category: string;
  note: string;
  tagIds: Set<string>;
}

interface SaveOperation {
  input: TaskAnnotationInput;
  revision: number;
}

function inputForDraft(draft: TaskDraft): TaskAnnotationInput {
  const category = draft.category.trim();
  return {
    category: category === "" ? null : category,
    note: draft.note,
    tag_ids: [...draft.tagIds].sort(),
  };
}

// A verdict-free, task-keyed note+tags editor. It reuses the example
// AnnotationEditor's debounced-autosave-with-single-inflight discipline so a
// durable per-task judgment survives across corpora and runs.
export function TaskAnnotationEditor({
  api,
  taskId,
  tags,
  onTagCreated,
}: {
  api: PreprocessingApi;
  taskId: string;
  tags: Tag[];
  onTagCreated: (tag: Tag) => void;
}) {
  const controlId = useId();
  const identity: TaskAnnotationIdentity = {
    dataset_id: datasetIdOf(taskId),
    task_id: taskId,
  };
  const [draft, setDraft] = useState<TaskDraft>({
    category: "",
    note: "",
    tagIds: new Set(),
  });
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [saveError, setSaveError] = useState("");
  const [newTag, setNewTag] = useState("");
  const [tagSaveState, setTagSaveState] = useState<"idle" | "saving" | "error">("idle");

  const activeRef = useRef(true);
  const debounceRef = useRef<number | undefined>(undefined);
  const draftRef = useRef(draft);
  const inFlightRef = useRef(false);
  const queuedRef = useRef<SaveOperation | null>(null);
  const revisionRef = useRef(0);
  const runQueueRef = useRef<() => void>(() => undefined);
  draftRef.current = draft;

  function updateSaveState(next: SaveState) {
    if (activeRef.current) setSaveState(next);
  }

  useEffect(() => {
    activeRef.current = true;
    setLoaded(false);
    setLoadError("");
    void api.getTaskAnnotation(identity).then(
      (annotation: TaskAnnotation | null) => {
        if (!activeRef.current) return;
        const next: TaskDraft = {
          category: annotation?.category ?? "",
          note: annotation?.note ?? "",
          tagIds: new Set(annotation?.tags.map(({ tag_id }) => tag_id) ?? []),
        };
        draftRef.current = next;
        setDraft(next);
        setSaveState("idle");
        setLoaded(true);
      },
      (error: unknown) => {
        if (activeRef.current) setLoadError(errorMessage(error));
      },
    );
    return () => {
      activeRef.current = false;
      if (debounceRef.current !== undefined) window.clearTimeout(debounceRef.current);
    };
    // Reloading whenever the task identity changes keys the editor to one task.
  }, [api, taskId]);

  runQueueRef.current = () => {
    if (inFlightRef.current || queuedRef.current === null) return;
    const operation = queuedRef.current;
    queuedRef.current = null;
    inFlightRef.current = true;
    void api.putTaskAnnotation(identity, operation.input).then(
      () => {
        inFlightRef.current = false;
        if (operation.revision === revisionRef.current) updateSaveState("saved");
        if (queuedRef.current !== null) runQueueRef.current();
      },
      (error: unknown) => {
        inFlightRef.current = false;
        if (queuedRef.current !== null) {
          runQueueRef.current();
          return;
        }
        updateSaveState("error");
        if (activeRef.current) setSaveError(errorMessage(error));
      },
    );
  };

  function queue(input: TaskAnnotationInput) {
    queuedRef.current = { input, revision: revisionRef.current };
    if (activeRef.current) setSaveError("");
    updateSaveState("saving");
    runQueueRef.current();
  }

  function updateDraft(next: TaskDraft, saveImmediately: boolean) {
    revisionRef.current += 1;
    draftRef.current = next;
    setDraft(next);
    setSaveError("");
    if (debounceRef.current !== undefined) {
      window.clearTimeout(debounceRef.current);
      debounceRef.current = undefined;
    }
    if (saveImmediately) {
      queue(inputForDraft(next));
      return;
    }
    updateSaveState("dirty");
    debounceRef.current = window.setTimeout(() => {
      debounceRef.current = undefined;
      queue(inputForDraft(draftRef.current));
    }, NOTE_AUTOSAVE_DELAY_MS);
  }

  function flushDraft() {
    if (debounceRef.current === undefined) return;
    window.clearTimeout(debounceRef.current);
    debounceRef.current = undefined;
    queue(inputForDraft(draftRef.current));
  }

  function toggleTag(tagId: string, checked: boolean) {
    const nextTagIds = new Set(draftRef.current.tagIds);
    if (checked) nextTagIds.add(tagId);
    else nextTagIds.delete(tagId);
    updateDraft({ ...draftRef.current, tagIds: nextTagIds }, true);
  }

  function createTag() {
    const name = newTag.trim();
    if (name === "") return;
    setTagSaveState("saving");
    void api.createTag(name).then(
      (tag) => {
        if (!activeRef.current) return;
        onTagCreated(tag);
        setNewTag("");
        setTagSaveState("idle");
        updateDraft(
          { ...draftRef.current, tagIds: new Set(draftRef.current.tagIds).add(tag.tag_id) },
          true,
        );
      },
      () => {
        if (activeRef.current) setTagSaveState("error");
      },
    );
  }

  return (
    <form className="annotation-editor task-annotation-editor" onSubmit={(event) => event.preventDefault()}>
      <div className="annotation-heading">
        <h3>Task judgment</h3>
        <div className={`save-state save-state--${saveState}`} aria-live="polite">
          {saveState === "dirty" && "Unsaved changes"}
          {saveState === "saving" && "Saving…"}
          {saveState === "saved" && "Saved"}
          {saveState === "error" && "Save failed"}
        </div>
      </div>
      <p className="task-annotation-identity">{taskId}</p>

      {loadError !== "" && <p className="inline-error" role="alert">Could not load task judgment: {loadError}</p>}

      <label className="field-label" htmlFor={`${controlId}-category`}>Category</label>
      <input
        disabled={!loaded}
        id={`${controlId}-category`}
        onBlur={flushDraft}
        onChange={(event) => updateDraft({ ...draftRef.current, category: event.target.value }, false)}
        placeholder="Optional label, e.g. difficulty or theme"
        value={draft.category}
      />

      <label className="field-label" htmlFor={`${controlId}-note`}>Note</label>
      <textarea
        disabled={!loaded}
        id={`${controlId}-note`}
        onBlur={flushDraft}
        onChange={(event) => updateDraft({ ...draftRef.current, note: event.target.value }, false)}
        placeholder="Durable judgment that survives across corpora and runs"
        rows={4}
        value={draft.note}
      />

      <div className="create-tag">
        <label htmlFor={`${controlId}-new-tag`}>Create task tag</label>
        <div>
          <input
            disabled={!loaded}
            id={`${controlId}-new-tag`}
            onChange={(event) => setNewTag(event.target.value)}
            placeholder="e.g. dynamic programming"
            value={newTag}
          />
          <button disabled={!loaded || newTag.trim() === "" || tagSaveState === "saving"} onClick={createTag} type="button">
            {tagSaveState === "error" ? "Retry create and select task tag" : "Create and select task tag"}
          </button>
        </div>
      </div>

      <fieldset className="tag-fieldset" disabled={!loaded}>
        <legend>Task tags</legend>
        {tags.length === 0 ? <p className="empty-state">No tags yet.</p> : (
          <div className="tag-options">
            {tags.map((tag) => (
              <label key={tag.tag_id}>
                <input
                  aria-label={`Task tag ${tag.name}`}
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
    </form>
  );
}
