import { useEffect, useRef, useState } from "react";

export const AUTOSAVE_DELAY_MS = 300;

export type SaveState = "idle" | "dirty" | "saving" | "saved" | "error";

interface AutosaveOptions<Input, Result> {
  onError?: (error: unknown) => void;
  onSaved?: (result: Result) => void;
  save: (input: Input) => Promise<Result>;
  scopeKey: string;
}

interface Operation<Input, Result> {
  generation: number;
  input: Input;
  onError?: (error: unknown) => void;
  onSaved?: (result: Result) => void;
  revision: number;
  save: (input: Input) => Promise<Result>;
}

interface DebouncedOperation<Input, Result> {
  operation: Operation<Input, Result>;
  timer: number;
}

interface Waiter {
  generation: number;
  resolve: (saved: boolean) => void;
}

export interface AutosaveQueue<Input> {
  currentSaveState: () => SaveState;
  edit: (input: Input, saveImmediately: boolean) => void;
  flush: () => Promise<boolean>;
  isUnsafe: () => boolean;
  saveState: SaveState;
}

/**
 * One race-safe autosave discipline shared by all annotation editors.
 *
 * Every operation captures its identity-scoped save callback. A scope switch
 * can therefore drain an older request without redirecting it to the new
 * identity, while generation and revision checks prevent old completions from
 * updating the current editor.
 */
export function useAutosaveQueue<Input, Result>(
  options: AutosaveOptions<Input, Result>,
): AutosaveQueue<Input> {
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const activeRef = useRef(true);
  const currentOptionsRef = useRef(options);
  const debouncedRef = useRef<DebouncedOperation<Input, Result> | null>(null);
  const failedRef = useRef<Operation<Input, Result> | null>(null);
  const generationRef = useRef(0);
  const inFlightRef = useRef<Operation<Input, Result> | null>(null);
  const queuedRef = useRef<Array<Operation<Input, Result>>>([]);
  const revisionRef = useRef(0);
  const runQueueRef = useRef<() => void>(() => undefined);
  const saveStateRef = useRef<SaveState>("idle");
  const scopeKeyRef = useRef(options.scopeKey);
  const waitersRef = useRef<Waiter[]>([]);
  currentOptionsRef.current = options;

  if (scopeKeyRef.current !== options.scopeKey) {
    scopeKeyRef.current = options.scopeKey;
    generationRef.current += 1;
    revisionRef.current = 0;
    failedRef.current = null;
    saveStateRef.current = "idle";
  }
  const renderedGeneration = generationRef.current;

  function updateSaveState(next: SaveState, generation: number) {
    if (generation !== generationRef.current) return;
    saveStateRef.current = next;
    if (activeRef.current) setSaveState(next);
  }

  function hasPending(generation: number): boolean {
    return (
      inFlightRef.current?.generation === generation
      || queuedRef.current.some((item) => item.generation === generation)
      || debouncedRef.current?.operation.generation === generation
    );
  }

  function settle(generation: number, saved: boolean) {
    if (hasPending(generation)) return;
    const remaining: Waiter[] = [];
    for (const waiter of waitersRef.current) {
      if (waiter.generation === generation) waiter.resolve(saved);
      else remaining.push(waiter);
    }
    waitersRef.current = remaining;
  }

  function enqueue(operation: Operation<Input, Result>) {
    const queuedIndex = queuedRef.current.findIndex(
      (item) => item.generation === operation.generation,
    );
    if (queuedIndex === -1) queuedRef.current.push(operation);
    else queuedRef.current[queuedIndex] = operation;
    updateSaveState("saving", operation.generation);
    runQueueRef.current();
  }

  runQueueRef.current = () => {
    if (inFlightRef.current !== null) return;
    const operation = queuedRef.current.shift();
    if (operation === undefined) return;
    inFlightRef.current = operation;
    void operation.save(operation.input).then(
      (result) => {
        inFlightRef.current = null;
        const isLatest = (
          operation.generation === generationRef.current
          && operation.revision === revisionRef.current
        );
        if (isLatest) {
          failedRef.current = null;
          updateSaveState("saved", operation.generation);
          if (activeRef.current) operation.onSaved?.(result);
        }
        runQueueRef.current();
        settle(operation.generation, true);
      },
      (error: unknown) => {
        inFlightRef.current = null;
        const newerWorkExists = (
          queuedRef.current.some(
            (item) => item.generation === operation.generation,
          )
          || debouncedRef.current?.operation.generation
            === operation.generation
        );
        const isLatest = (
          operation.generation === generationRef.current
          && operation.revision === revisionRef.current
        );
        if (isLatest && !newerWorkExists) {
          failedRef.current = operation;
          updateSaveState("error", operation.generation);
          if (activeRef.current) operation.onError?.(error);
        }
        runQueueRef.current();
        settle(operation.generation, false);
      },
    );
  };

  function operation(input: Input): Operation<Input, Result> {
    const current = currentOptionsRef.current;
    return {
      generation: generationRef.current,
      input,
      onError: current.onError,
      onSaved: current.onSaved,
      revision: revisionRef.current,
      save: current.save,
    };
  }

  function clearDebounce(generation: number): Operation<Input, Result> | null {
    const debounced = debouncedRef.current;
    if (
      debounced === null
      || debounced.operation.generation !== generation
    ) {
      return null;
    }
    window.clearTimeout(debounced.timer);
    debouncedRef.current = null;
    return debounced.operation;
  }

  function flushGeneration(generation: number): Promise<boolean> {
    const pendingDraft = clearDebounce(generation);
    if (pendingDraft !== null) enqueue(pendingDraft);
    const failed = failedRef.current;
    if (
      failed?.generation === generation
      && !hasPending(generation)
    ) {
      failedRef.current = null;
      enqueue(failed);
    }
    if (!hasPending(generation)) {
      return Promise.resolve(
        !(
          failedRef.current?.generation === generation
          || (
            generation === generationRef.current
            && saveStateRef.current === "error"
          )
        ),
      );
    }
    return new Promise((resolve) => {
      waitersRef.current.push({ generation, resolve });
    });
  }

  useEffect(() => {
    activeRef.current = true;
    setSaveState(saveStateRef.current);
    return () => {
      if (generationRef.current === renderedGeneration) {
        activeRef.current = false;
      }
      void flushGeneration(renderedGeneration);
    };
  }, [options.scopeKey, renderedGeneration]);

  return {
    currentSaveState: () => saveStateRef.current,
    edit(input: Input, saveImmediately: boolean) {
      revisionRef.current += 1;
      failedRef.current = null;
      clearDebounce(generationRef.current);
      const next = operation(input);
      if (saveImmediately) {
        enqueue(next);
        return;
      }
      updateSaveState("dirty", next.generation);
      debouncedRef.current = {
        operation: next,
        timer: window.setTimeout(() => {
          if (debouncedRef.current?.operation !== next) return;
          debouncedRef.current = null;
          enqueue(next);
        }, AUTOSAVE_DELAY_MS),
      };
    },
    flush: () => flushGeneration(generationRef.current),
    isUnsafe: () => (
      hasPending(generationRef.current)
      || failedRef.current?.generation === generationRef.current
      || ["dirty", "saving", "error"].includes(saveStateRef.current)
    ),
    saveState,
  };
}
