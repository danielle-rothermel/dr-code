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
import { useAutosaveQueue } from "./use-autosave-queue";

interface TaskDraft {
  category: string;
  note: string;
  tagIds: Set<string>;
}

function inputForDraft(draft: TaskDraft): TaskAnnotationInput {
  const category = draft.category.trim();
  return {
    category: category === "" ? null : category,
    note: draft.note,
    tag_ids: [...draft.tagIds].sort(),
  };
}

// A verdict-free, task-keyed note+tags editor. It shares the example
// AnnotationEditor's debounced-autosave-with-single-inflight discipline via
// useAutosaveQueue so a durable per-task judgment survives across corpora and
// runs.
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
  const [saveError, setSaveError] = useState("");
  const [newTag, setNewTag] = useState("");
  const [tagSaveState, setTagSaveState] = useState<"idle" | "saving" | "error">("idle");

  const activeRef = useRef(true);
  const draftRef = useRef(draft);
  draftRef.current = draft;

  const autosave = useAutosaveQueue<TaskAnnotationInput, TaskAnnotation>({
    save: (input) => api.putTaskAnnotation(identity, input),
    onSaved: () => undefined,
    onError: (_revision, error) => {
      if (activeRef.current) setSaveError(errorMessage(error));
    },
  });

  useEffect(() => {
    activeRef.current = true;
    autosave.markActive(true);
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
        setLoaded(true);
      },
      (error: unknown) => {
        if (activeRef.current) setLoadError(errorMessage(error));
      },
    );
    return () => {
      activeRef.current = false;
      autosave.markActive(false);
    };
    // Reloading whenever the task identity changes keys the editor to one task.
  }, [api, taskId]);

  function updateDraft(next: TaskDraft, saveImmediately: boolean) {
    draftRef.current = next;
    setDraft(next);
    setSaveError("");
    autosave.edit(inputForDraft(next), saveImmediately);
  }

  function flushDraft() {
    autosave.flush(inputForDraft(draftRef.current));
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

  const saveState = autosave.saveState;

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
