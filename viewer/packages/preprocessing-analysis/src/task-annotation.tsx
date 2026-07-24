import { useEffect, useId, useRef, useState } from "react";

import {
  TASK_CATEGORY_MAX_LENGTH,
  TASK_NOTE_MAX_LENGTH,
  TASK_TAG_IDS_MAX_ITEMS,
} from "./api";
import type {
  PreprocessingApi,
  Tag,
  TaskAnnotation,
  TaskAnnotationIdentity,
  TaskAnnotationInput,
} from "./api";
import { errorMessage } from "./format";
import type { RegisterCardGuard } from "./review";
import { useAutosaveQueue } from "./use-autosave-queue";

interface TaskDraft {
  category: string;
  note: string;
  tagIds: Set<string>;
}

type TaskAnnotationMutation =
  | { input: TaskAnnotationInput; kind: "put" }
  | { kind: "delete" };

function inputForDraft(draft: TaskDraft): TaskAnnotationInput {
  const category = draft.category.trim();
  return {
    category: category === "" ? null : category,
    note: draft.note,
    tag_ids: [...draft.tagIds].sort(),
  };
}

export function TaskAnnotationEditor({
  api,
  onTagCreated,
  registerCardGuard,
  tags,
  identity,
}: {
  api: PreprocessingApi;
  onTagCreated: (tag: Tag) => void;
  registerCardGuard: RegisterCardGuard;
  tags: Tag[];
  identity: TaskAnnotationIdentity;
}) {
  const controlId = useId();
  const identityKey = JSON.stringify([
    identity.dataset_id,
    identity.task_id,
    identity.task_identity,
  ]);
  const [draft, setDraft] = useState<TaskDraft>({
    category: "",
    note: "",
    tagIds: new Set(),
  });
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [loadRevision, setLoadRevision] = useState(0);
  const [saveError, setSaveError] = useState("");
  const [loadedMachine, setLoadedMachine] = useState(false);
  const [hasSavedAnnotation, setHasSavedAnnotation] = useState(false);
  const [newTag, setNewTag] = useState("");
  const [tagError, setTagError] = useState("");
  const [tagSaving, setTagSaving] = useState(false);
  const activeRef = useRef(true);
  const draftRef = useRef(draft);
  const newTagRef = useRef(newTag);
  const tagErrorRef = useRef(tagError);
  const pendingTagRef = useRef<Promise<boolean> | null>(null);
  draftRef.current = draft;
  newTagRef.current = newTag;
  tagErrorRef.current = tagError;

  const autosave = useAutosaveQueue<TaskAnnotationMutation, TaskAnnotation | null>({
    onError: (error) => {
      if (activeRef.current) setSaveError(errorMessage(error));
    },
    onSaved: (annotation) => {
      if (!activeRef.current) return;
      if (annotation === null) {
        const emptyDraft: TaskDraft = {
          category: "",
          note: "",
          tagIds: new Set(),
        };
        draftRef.current = emptyDraft;
        setDraft(emptyDraft);
        setHasSavedAnnotation(false);
        setLoadedMachine(false);
        return;
      }
      setHasSavedAnnotation(true);
      setLoadedMachine(false);
      const nextDraft = {
        category: annotation.category ?? "",
        note: annotation.note ?? "",
        tagIds: new Set(annotation.tags.map(({ tag_id }) => tag_id)),
      };
      draftRef.current = nextDraft;
      setDraft(nextDraft);
    },
    save: async (mutation) => {
      if (mutation.kind === "delete") {
        await api.deleteTaskAnnotation(identity);
        return null;
      }
      return api.putTaskAnnotation(identity, mutation.input);
    },
    scopeKey: identityKey,
  });

  useEffect(() => {
    let current = true;
    activeRef.current = true;
    setLoaded(false);
    setLoadError("");
    setSaveError("");
    setLoadedMachine(false);
    setHasSavedAnnotation(false);
    void api.getTaskAnnotation(identity).then(
      (annotation) => {
        if (!current || !activeRef.current) return;
        const next = {
          category: annotation?.category ?? "",
          note: annotation?.note ?? "",
          tagIds: new Set(
            annotation?.tags.map(({ tag_id }) => tag_id) ?? [],
          ),
        };
        draftRef.current = next;
        setDraft(next);
        setLoadedMachine(annotation?.origin === "machine");
        setHasSavedAnnotation(annotation !== null);
        setLoaded(true);
      },
      (error: unknown) => {
        if (current && activeRef.current) {
          setLoadError(errorMessage(error));
        }
      },
    );
    return () => {
      current = false;
      activeRef.current = false;
    };
  }, [api, identityKey, loadRevision]);

  function updateDraft(next: TaskDraft, saveImmediately: boolean) {
    draftRef.current = next;
    setDraft(next);
    setSaveError("");
    autosave.edit({ input: inputForDraft(next), kind: "put" }, saveImmediately);
  }

  function toggleTag(tagId: string, checked: boolean) {
    const nextTags = new Set(draftRef.current.tagIds);
    if (checked && nextTags.size < TASK_TAG_IDS_MAX_ITEMS) {
      nextTags.add(tagId);
    }
    else nextTags.delete(tagId);
    updateDraft({ ...draftRef.current, tagIds: nextTags }, true);
  }

  function deleteSavedAnnotation() {
    if (!hasSavedAnnotation || pendingTagRef.current !== null) return;
    discardTagCreationIntent();
    setSaveError("");
    autosave.edit({ kind: "delete" }, true);
  }

  function updateTagError(error: string) {
    tagErrorRef.current = error;
    if (activeRef.current) setTagError(error);
  }

  function discardTagCreationIntent() {
    newTagRef.current = "";
    updateTagError("");
    if (activeRef.current) {
      setNewTag("");
      setTagSaving(false);
    }
  }

  function startTagCreation(name: string): Promise<boolean> {
    if (draftRef.current.tagIds.size >= TASK_TAG_IDS_MAX_ITEMS) {
      return Promise.resolve(false);
    }
    if (pendingTagRef.current !== null) return pendingTagRef.current;
    setTagSaving(true);
    updateTagError("");
    const request = api.createTag(name).then(
      (tag) => {
        if (!activeRef.current) return false;
        onTagCreated(tag);
        setNewTag("");
        setTagSaving(false);
        updateDraft(
          {
            ...draftRef.current,
            tagIds: new Set(draftRef.current.tagIds).add(tag.tag_id),
          },
          true,
        );
        return true;
      },
      (error: unknown) => {
        if (activeRef.current) {
          setTagSaving(false);
        }
        updateTagError(errorMessage(error));
        return false;
      },
    ).finally(() => {
      pendingTagRef.current = null;
    });
    pendingTagRef.current = request;
    return request;
  }

  async function flush(): Promise<boolean> {
    if (pendingTagRef.current !== null && !await pendingTagRef.current) {
      return false;
    }
    if (tagErrorRef.current !== "") {
      const name = newTagRef.current.trim();
      if (name === "" || !await startTagCreation(name)) return false;
    }
    return autosave.flush();
  }

  useEffect(
    () => registerCardGuard(
      `task:${identityKey}`,
      {
        flush,
        isUnsafe: () => (
          pendingTagRef.current !== null
          || tagErrorRef.current !== ""
          || autosave.isUnsafe()
        ),
      },
    ),
    [identityKey, registerCardGuard, tagError],
  );

  return (
    <form
      className="annotation-editor task-annotation-editor"
      onSubmit={(event) => event.preventDefault()}
    >
      <div className="annotation-heading">
        <h3>Task annotation</h3>
        <div
          aria-live="polite"
          className={`save-state save-state--${autosave.saveState}`}
        >
          {autosave.saveState === "dirty" && "Unsaved changes"}
          {autosave.saveState === "saving" && "Saving…"}
          {autosave.saveState === "saved" && "Saved"}
          {autosave.saveState === "error" && "Save failed"}
        </div>
      </div>
      <p className="task-annotation-identity">
        {identity.dataset_id} · {identity.task_id}
      </p>

      {loadError !== "" && (
        <div className="annotation-load-error">
          <p className="inline-error" role="alert">
            Could not load task annotation: {loadError}
          </p>
          <button
            onClick={() => setLoadRevision((revision) => revision + 1)}
            type="button"
          >
            Retry loading task annotation
          </button>
        </div>
      )}
      {loadedMachine && (
        <p className="machine-annotation-notice">
          This machine annotation includes provenance. Editing any field saves
          it as human review and clears that provenance.
        </p>
      )}

      <label className="field-label" htmlFor={`${controlId}-task-category`}>
        Category
      </label>
      <input
        disabled={!loaded}
        id={`${controlId}-task-category`}
        maxLength={TASK_CATEGORY_MAX_LENGTH}
        onBlur={() => void autosave.flush()}
        onChange={(event) => updateDraft(
          { ...draftRef.current, category: event.target.value },
          false,
        )}
        placeholder="Optional task category"
        value={draft.category}
      />

      <label className="field-label" htmlFor={`${controlId}-task-note`}>
        Task note
      </label>
      <textarea
        disabled={!loaded}
        id={`${controlId}-task-note`}
        maxLength={TASK_NOTE_MAX_LENGTH}
        onBlur={() => void autosave.flush()}
        onChange={(event) => updateDraft(
          { ...draftRef.current, note: event.target.value },
          false,
        )}
        placeholder="Judgment shared across runs"
        rows={4}
        value={draft.note}
      />

      <div className="create-tag">
        <label htmlFor={`${controlId}-task-tag`}>Create task tag</label>
        <div>
          <input
            disabled={!loaded}
            id={`${controlId}-task-tag`}
            onChange={(event) => setNewTag(event.target.value)}
            placeholder="e.g. dynamic programming"
            value={newTag}
          />
          <button
            disabled={
              !loaded
              || newTag.trim() === ""
              || tagSaving
              || draft.tagIds.size >= TASK_TAG_IDS_MAX_ITEMS
            }
            onClick={() => void startTagCreation(newTag.trim())}
            type="button"
          >
            {tagError === ""
              ? "Create and select task tag"
              : "Retry create and select task tag"}
          </button>
        </div>
      </div>

      <fieldset className="tag-fieldset" disabled={!loaded}>
        <legend>Task tags</legend>
        {tags.length === 0 ? (
          <p className="empty-state">No tags yet.</p>
        ) : (
          <div className="tag-options">
            {tags.map((tag) => (
              <label key={tag.tag_id}>
                <input
                  aria-label={`Task tag ${tag.name}`}
                  checked={draft.tagIds.has(tag.tag_id)}
                  disabled={
                    !draft.tagIds.has(tag.tag_id)
                    && draft.tagIds.size >= TASK_TAG_IDS_MAX_ITEMS
                  }
                  onChange={(event) => toggleTag(
                    tag.tag_id,
                    event.target.checked,
                  )}
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
          !loaded
          || !hasSavedAnnotation
          || autosave.currentSaveState() === "saving"
        }
        onClick={deleteSavedAnnotation}
        type="button"
      >
        Delete task annotation
      </button>

      {saveError !== "" && (
        <p className="inline-error" role="alert">{saveError}</p>
      )}
      {tagError !== "" && (
        <p className="inline-error" role="alert">
          Task tag creation failed: {tagError}
        </p>
      )}
    </form>
  );
}
