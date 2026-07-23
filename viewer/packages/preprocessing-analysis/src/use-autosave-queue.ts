import { useRef, useState } from "react";

export const NOTE_AUTOSAVE_DELAY_MS = 300;

export type SaveState = "idle" | "dirty" | "saving" | "saved" | "error";

// The single source of truth for the debounced, single-inflight autosave
// discipline shared by every annotation editor. One save runs at a time; while
// it is in flight the newest draft is coalesced into a single queued operation.
// A monotonic revision guards against a resolved save clobbering newer edits, so
// the terminal "saved"/"error" transition only lands when the operation still
// reflects the latest draft.
interface AutosaveCallbacks<Input, Result> {
  // Persist one queued input. Rejections drive the error path below.
  save: (input: Input) => Promise<Result>;
  // The queued operation is still the latest edit. Runs on the success path.
  onSaved: (revision: number, result: Result) => void;
  // The queued operation is still the latest edit. Runs on the error path.
  onError: (revision: number, error: unknown) => void;
  // The queue has drained with nothing debounced. `succeeded` reports whether the
  // last operation persisted, letting callers settle any leave/flush waiters.
  onIdle?: (succeeded: boolean) => void;
}

interface QueuedOperation<Input> {
  input: Input;
  revision: number;
}

export interface AutosaveQueue<Input> {
  saveState: SaveState;
  // Report an edit. `saveImmediately` bypasses the debounce (discrete choices
  // such as a verdict or tag toggle); otherwise the input is debounced.
  edit: (input: Input, saveImmediately: boolean) => void;
  // Persist any debounced-but-unsent draft now (e.g. on blur or before leaving).
  flush: (input: Input) => void;
  // Requeue `input` only if nothing is in flight or already queued; used to retry
  // after an error before leaving the card.
  requeueIfSettled: (input: Input) => void;
  // Synchronous view of the terminal save state for guards that run outside
  // render (before-leave flush, beforeunload).
  currentSaveState: () => SaveState;
  // True when no save is in flight and none is queued.
  isSettled: () => boolean;
  // Toggle the mounted flag so late resolutions do not touch unmounted state.
  markActive: (active: boolean) => void;
}

export function useAutosaveQueue<Input, Result>(
  callbacks: AutosaveCallbacks<Input, Result>,
): AutosaveQueue<Input> {
  const [saveState, setSaveState] = useState<SaveState>("idle");

  const activeRef = useRef(true);
  const callbacksRef = useRef(callbacks);
  const debounceRef = useRef<number | undefined>(undefined);
  const inFlightRef = useRef(false);
  const queuedRef = useRef<QueuedOperation<Input> | null>(null);
  const revisionRef = useRef(0);
  const runQueueRef = useRef<() => void>(() => undefined);
  const saveStateRef = useRef(saveState);
  callbacksRef.current = callbacks;

  function updateSaveState(next: SaveState) {
    saveStateRef.current = next;
    if (activeRef.current) setSaveState(next);
  }

  runQueueRef.current = () => {
    if (inFlightRef.current || queuedRef.current === null) return;
    const operation = queuedRef.current;
    queuedRef.current = null;
    inFlightRef.current = true;
    void callbacksRef.current.save(operation.input).then(
      (result) => {
        inFlightRef.current = false;
        if (operation.revision === revisionRef.current) {
          updateSaveState("saved");
          callbacksRef.current.onSaved(operation.revision, result);
        }
        if (queuedRef.current !== null) {
          runQueueRef.current();
          return;
        }
        if (debounceRef.current === undefined) callbacksRef.current.onIdle?.(true);
      },
      (error: unknown) => {
        inFlightRef.current = false;
        if (queuedRef.current !== null) {
          runQueueRef.current();
          return;
        }
        // Suppress a stale rejection: a newer edit is already debounced and will
        // drive its own terminal transition, so this failure is no longer shown.
        if (operation.revision !== revisionRef.current && debounceRef.current !== undefined) return;
        updateSaveState("error");
        callbacksRef.current.onError(operation.revision, error);
        callbacksRef.current.onIdle?.(false);
      },
    );
  };

  function queue(input: Input) {
    queuedRef.current = { input, revision: revisionRef.current };
    updateSaveState("saving");
    runQueueRef.current();
  }

  function clearDebounce(): boolean {
    if (debounceRef.current === undefined) return false;
    window.clearTimeout(debounceRef.current);
    debounceRef.current = undefined;
    return true;
  }

  function edit(input: Input, saveImmediately: boolean) {
    revisionRef.current += 1;
    clearDebounce();
    if (saveImmediately) {
      queue(input);
      return;
    }
    updateSaveState("dirty");
    debounceRef.current = window.setTimeout(() => {
      debounceRef.current = undefined;
      queue(input);
    }, NOTE_AUTOSAVE_DELAY_MS);
  }

  return {
    saveState,
    edit,
    flush(input: Input) {
      if (clearDebounce()) queue(input);
    },
    requeueIfSettled(input: Input) {
      if (!inFlightRef.current && queuedRef.current === null) queue(input);
    },
    currentSaveState: () => saveStateRef.current,
    isSettled: () => !inFlightRef.current && queuedRef.current === null,
    markActive(active: boolean) {
      activeRef.current = active;
    },
  };
}
