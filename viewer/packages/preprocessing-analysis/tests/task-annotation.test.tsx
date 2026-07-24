import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { TaskAnnotation } from "../src/api";
import { TaskAnnotationEditor } from "../src/task-annotation";
import { fakeApi } from "./fixtures";

afterEach(cleanup);

const DATASET_ID = "evalplus/humanevalplus";

function identity(taskId: string) {
  return {
    dataset_id: DATASET_ID,
    task_id: taskId,
    task_identity: (taskId.endsWith("/43") ? "b" : "a").repeat(64),
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, reject, resolve };
}

function taskAnnotation(
  taskId: string,
  overrides: Partial<TaskAnnotation> = {},
): TaskAnnotation {
  return {
    category: null,
    identity: identity(taskId),
    note: null,
    origin: "human",
    provenance: null,
    tags: [],
    ...overrides,
  };
}

const registerCardGuard = vi.fn(() => () => undefined);

function editor(
  api = fakeApi(),
  taskId = "HumanEval/42",
) {
  return (
    <TaskAnnotationEditor
      api={api}
      identity={identity(taskId)}
      onTagCreated={vi.fn()}
      registerCardGuard={registerCardGuard}
      tags={[]}
    />
  );
}

describe("TaskAnnotationEditor", () => {
  it("loads machine provenance, then writes only human-editable fields", async () => {
    const machine = taskAnnotation("HumanEval/42", {
      category: "machine-category",
      note: "machine note",
      origin: "machine",
      provenance: {
        agreement: 0.75,
        extra: { repeats_agreed: 3 },
        model: "review-model",
        repeats: 4,
        taxonomy_version: "v2",
      },
    });
    const putTaskAnnotation = vi.fn(async (identity, input) => ({
      ...machine,
      category: input.category,
      identity,
      note: input.note,
      origin: "human" as const,
      provenance: null,
    }));
    const api = fakeApi({
      getTaskAnnotation: vi.fn().mockResolvedValue(machine),
      putTaskAnnotation,
    });
    render(editor(api));

    expect(await screen.findByText(/clears that provenance/)).toBeTruthy();
    const category = screen.getByLabelText("Category");
    fireEvent.change(category, { target: { value: "human-category" } });
    fireEvent.blur(category);

    await waitFor(() => expect(putTaskAnnotation).toHaveBeenCalledWith(
      identity("HumanEval/42"),
      {
        category: "human-category",
        note: "machine note",
        tag_ids: [],
      },
    ));
    await screen.findByText("Saved");
    expect(screen.queryByText(/clears that provenance/)).toBeNull();
  });

  it("deletes a human override and restores the absent task-annotation state", async () => {
    const machine = taskAnnotation("HumanEval/42", {
      category: "machine-category",
      note: "machine note",
      origin: "machine",
      provenance: {
        agreement: 0.75,
        extra: { repeats_agreed: 3 },
        model: "review-model",
        repeats: 4,
        taxonomy_version: "v2",
      },
    });
    const putTaskAnnotation = vi.fn(async (identity, input) => taskAnnotation(
      identity.task_id,
      {
        category: input.category,
        identity,
        note: input.note,
      },
    ));
    const deleteTaskAnnotation = vi.fn().mockResolvedValue(undefined);
    const api = fakeApi({
      deleteTaskAnnotation,
      getTaskAnnotation: vi.fn().mockResolvedValue(machine),
      putTaskAnnotation,
    });
    render(editor(api));

    const category = await screen.findByLabelText("Category");
    fireEvent.change(category, { target: { value: "human-category" } });
    fireEvent.blur(category);
    await screen.findByText("Saved");
    expect(screen.getByRole("button", { name: "Delete task annotation" }))
      .toBeTruthy();

    fireEvent.click(screen.getByRole("button", {
      name: "Delete task annotation",
    }));
    await waitFor(() => expect(deleteTaskAnnotation).toHaveBeenCalledWith(
      identity("HumanEval/42"),
    ));
    await waitFor(() => expect(
      (screen.getByLabelText("Category") as HTMLInputElement).value,
    ).toBe(""));

    expect((screen.getByLabelText("Category") as HTMLInputElement).value)
      .toBe("");
    expect((screen.getByLabelText("Task note") as HTMLTextAreaElement).value)
      .toBe("");
    expect(screen.queryByText(/clears that provenance/)).toBeNull();
    expect((screen.getByRole("button", {
      name: "Delete task annotation",
    }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("keeps a failed delete visible and blocks navigation until it recovers", async () => {
    const annotation = taskAnnotation("HumanEval/42", { note: "human note" });
    const deleteTaskAnnotation = vi.fn()
      .mockRejectedValueOnce(new Error("database unavailable"))
      .mockResolvedValueOnce(undefined);
    const api = fakeApi({
      deleteTaskAnnotation,
      getTaskAnnotation: vi.fn().mockResolvedValue(annotation),
    });
    let guard: {
      flush: () => Promise<boolean>;
      isUnsafe: () => boolean;
    } | undefined;
    const register = vi.fn((_key, nextGuard) => {
      guard = nextGuard;
      return () => undefined;
    });
    render(
      <TaskAnnotationEditor
        api={api}
        identity={identity("HumanEval/42")}
        onTagCreated={vi.fn()}
        registerCardGuard={register}
        tags={[]}
      />,
    );

    await screen.findByDisplayValue("human note");
    fireEvent.click(screen.getByRole("button", {
      name: "Delete task annotation",
    }));
    expect(await screen.findByText("Save failed")).toBeTruthy();
    expect(await screen.findByText("database unavailable")).toBeTruthy();
    expect((screen.getByLabelText("Task note") as HTMLTextAreaElement).value)
      .toBe("human note");
    expect(guard?.isUnsafe()).toBe(true);

    let canLeave = false;
    await act(async () => {
      canLeave = await guard!.flush();
    });
    expect(canLeave).toBe(true);
    expect(deleteTaskAnnotation).toHaveBeenCalledTimes(2);
    expect((screen.getByLabelText("Task note") as HTMLTextAreaElement).value)
      .toBe("");
    expect(guard?.isUnsafe()).toBe(false);
  });

  it("coalesces rapid revisions behind one pending save without stale overwrite", async () => {
    const first = deferred<TaskAnnotation>();
    const latest = deferred<TaskAnnotation>();
    const putTaskAnnotation = vi.fn()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(latest.promise);
    const api = fakeApi({
      getTaskAnnotation: vi.fn().mockResolvedValue(null),
      putTaskAnnotation,
    });
    render(editor(api));
    const note = await screen.findByLabelText("Task note");

    fireEvent.change(note, { target: { value: "first" } });
    fireEvent.blur(note);
    expect(putTaskAnnotation).toHaveBeenCalledTimes(1);
    fireEvent.change(note, { target: { value: "second" } });
    fireEvent.blur(note);
    fireEvent.change(note, { target: { value: "latest" } });
    fireEvent.blur(note);
    expect(putTaskAnnotation).toHaveBeenCalledTimes(1);

    await act(async () => {
      first.resolve(taskAnnotation("HumanEval/42", { note: "first" }));
      await first.promise;
    });
    expect((screen.getByLabelText("Task note") as HTMLTextAreaElement).value)
      .toBe("latest");
    expect(putTaskAnnotation).toHaveBeenCalledTimes(2);
    expect(putTaskAnnotation).toHaveBeenLastCalledWith(
      expect.anything(),
      expect.objectContaining({ note: "latest" }),
    );

    await act(async () => {
      latest.resolve(taskAnnotation("HumanEval/42", { note: "latest" }));
      await latest.promise;
    });
    await screen.findByText("Saved");
  });

  it("preserves a failed latest draft and retries the newer edit", async () => {
    const putTaskAnnotation = vi.fn()
      .mockRejectedValueOnce(new Error("database unavailable"))
      .mockResolvedValueOnce(taskAnnotation("HumanEval/42", {
        note: "newer",
      }));
    const api = fakeApi({
      getTaskAnnotation: vi.fn().mockResolvedValue(null),
      putTaskAnnotation,
    });
    render(editor(api));
    const note = await screen.findByLabelText("Task note");

    fireEvent.change(note, { target: { value: "failed draft" } });
    fireEvent.blur(note);
    expect(await screen.findByText("Save failed")).toBeTruthy();
    expect((note as HTMLTextAreaElement).value).toBe("failed draft");

    fireEvent.change(note, { target: { value: "newer" } });
    fireEvent.blur(note);
    await screen.findByText("Saved");
    expect(putTaskAnnotation).toHaveBeenLastCalledWith(
      expect.anything(),
      expect.objectContaining({ note: "newer" }),
    );
  });

  it("isolates late completion from a switched identity", async () => {
    const oldSave = deferred<TaskAnnotation>();
    const putTaskAnnotation = vi.fn().mockReturnValue(oldSave.promise);
    const getTaskAnnotation = vi.fn(async (identity) => (
      identity.task_id === "HumanEval/43"
        ? taskAnnotation("HumanEval/43", { note: "new identity" })
        : null
    ));
    const api = fakeApi({ getTaskAnnotation, putTaskAnnotation });
    const view = render(editor(api));
    const note = await screen.findByLabelText("Task note");
    fireEvent.change(note, { target: { value: "old draft" } });
    fireEvent.blur(note);

    view.rerender(editor(api, "HumanEval/43"));
    await waitFor(() => expect(
      (screen.getByLabelText("Task note") as HTMLTextAreaElement).value,
    ).toBe("new identity"));
    await act(async () => {
      oldSave.resolve(taskAnnotation("HumanEval/42", { note: "old draft" }));
      await oldSave.promise;
    });
    expect((screen.getByLabelText("Task note") as HTMLTextAreaElement).value)
      .toBe("new identity");
  });

  it("reports an old generation rejection to its pending flush", async () => {
    const oldSave = deferred<TaskAnnotation>();
    const putTaskAnnotation = vi.fn().mockReturnValue(oldSave.promise);
    const getTaskAnnotation = vi.fn().mockResolvedValue(null);
    const api = fakeApi({ getTaskAnnotation, putTaskAnnotation });
    const guards = new Map<string, {
      flush: () => Promise<boolean>;
      isUnsafe: () => boolean;
    }>();
    const register = vi.fn((key, guard) => {
      guards.set(key, guard);
      return () => {
        guards.delete(key);
      };
    });
    const renderEditor = (taskId: string) => (
      <TaskAnnotationEditor
        api={api}
        identity={identity(taskId)}
        onTagCreated={vi.fn()}
        registerCardGuard={register}
        tags={[]}
      />
    );
    const view = render(renderEditor("HumanEval/42"));
    const note = await screen.findByLabelText("Task note");
    fireEvent.change(note, { target: { value: "must not be lost" } });
    fireEvent.blur(note);
    const oldKey = `task:${JSON.stringify([
      DATASET_ID,
      "HumanEval/42",
      "a".repeat(64),
    ])}`;
    const flush = guards.get(oldKey)?.flush();
    expect(flush).toBeDefined();

    view.rerender(renderEditor("HumanEval/43"));
    await waitFor(() => expect(getTaskAnnotation).toHaveBeenCalledTimes(2));
    await act(async () => {
      oldSave.reject(new Error("old generation rejected"));
      await oldSave.promise.catch(() => undefined);
    });

    await expect(flush).resolves.toBe(false);
  });

  it("can retry a failed annotation load", async () => {
    const getTaskAnnotation = vi.fn()
      .mockRejectedValueOnce(new Error("service unavailable"))
      .mockResolvedValueOnce(taskAnnotation("HumanEval/42", {
        note: "loaded after retry",
      }));
    render(editor(fakeApi({ getTaskAnnotation })));

    expect(await screen.findByText(
      /Could not load task annotation: service unavailable/,
    )).toBeTruthy();
    expect((screen.getByLabelText("Task note") as HTMLTextAreaElement).disabled)
      .toBe(true);
    fireEvent.click(screen.getByRole("button", {
      name: "Retry loading task annotation",
    }));

    await waitFor(() => expect(getTaskAnnotation).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(
      (screen.getByLabelText("Task note") as HTMLTextAreaElement).value,
    ).toBe("loaded after retry"));
    expect((screen.getByLabelText("Task note") as HTMLTextAreaElement).disabled)
      .toBe(false);
  });

  it("retries failed tag creation with the latest edited name", async () => {
    const correctedTag = { name: "corrected", tag_id: "corrected-tag" };
    const createTag = vi.fn()
      .mockRejectedValueOnce(new Error("tag service unavailable"))
      .mockResolvedValueOnce(correctedTag);
    const putTaskAnnotation = vi.fn(async (identity, input) => taskAnnotation(
      identity.task_id,
      {
        identity,
        tags: input.tag_ids.map((tagId: string) => (
          tagId === correctedTag.tag_id ? correctedTag : {
            name: tagId,
            tag_id: tagId,
          }
        )),
      },
    ));
    const api = fakeApi({
      createTag,
      getTaskAnnotation: vi.fn().mockResolvedValue(null),
      putTaskAnnotation,
    });
    let guard: {
      flush: () => Promise<boolean>;
      isUnsafe: () => boolean;
    } | undefined;
    const register = vi.fn((_key, nextGuard) => {
      guard = nextGuard;
      return () => undefined;
    });
    render(
      <TaskAnnotationEditor
        api={api}
        identity={identity("HumanEval/42")}
        onTagCreated={vi.fn()}
        registerCardGuard={register}
        tags={[]}
      />,
    );
    const tagName = await screen.findByLabelText("Create task tag");
    fireEvent.change(tagName, { target: { value: "old" } });
    fireEvent.click(screen.getByRole("button", {
      name: "Create and select task tag",
    }));
    expect(await screen.findByText(
      /Task tag creation failed: tag service unavailable/,
    )).toBeTruthy();

    fireEvent.change(tagName, { target: { value: "corrected" } });
    expect(guard).toBeDefined();
    let saved = false;
    await act(async () => {
      saved = await guard!.flush();
    });

    expect(saved).toBe(true);
    expect(createTag.mock.calls.map(([name]) => name)).toEqual([
      "old",
      "corrected",
    ]);
    expect(putTaskAnnotation).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ tag_ids: [correctedTag.tag_id] }),
    );
  });

  it("discards failed tag intent before delete so navigation cannot resurrect it", async () => {
    const createTag = vi.fn().mockRejectedValue(
      new Error("tag service unavailable"),
    );
    const deleteTaskAnnotation = vi.fn().mockResolvedValue(undefined);
    const putTaskAnnotation = vi.fn();
    const api = fakeApi({
      createTag,
      deleteTaskAnnotation,
      getTaskAnnotation: vi.fn().mockResolvedValue(
        taskAnnotation("HumanEval/42", { note: "delete me" }),
      ),
      putTaskAnnotation,
    });
    let guard: {
      flush: () => Promise<boolean>;
      isUnsafe: () => boolean;
    } | undefined;
    const register = vi.fn((_key, nextGuard) => {
      guard = nextGuard;
      return () => undefined;
    });
    render(
      <TaskAnnotationEditor
        api={api}
        identity={identity("HumanEval/42")}
        onTagCreated={vi.fn()}
        registerCardGuard={register}
        tags={[]}
      />,
    );

    await screen.findByDisplayValue("delete me");
    fireEvent.change(screen.getByLabelText("Create task tag"), {
      target: { value: "must not retry" },
    });
    fireEvent.click(screen.getByRole("button", {
      name: "Create and select task tag",
    }));
    expect(await screen.findByText(
      /Task tag creation failed: tag service unavailable/,
    )).toBeTruthy();

    fireEvent.click(screen.getByRole("button", {
      name: "Delete task annotation",
    }));
    await waitFor(() => expect(deleteTaskAnnotation).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(
      (screen.getByLabelText("Task note") as HTMLTextAreaElement).value,
    ).toBe(""));

    let canNavigate = false;
    await act(async () => {
      canNavigate = await guard!.flush();
    });
    expect(canNavigate).toBe(true);
    expect(createTag).toHaveBeenCalledTimes(1);
    expect(putTaskAnnotation).not.toHaveBeenCalled();
    expect(deleteTaskAnnotation).toHaveBeenCalledTimes(1);
    expect(screen.queryByText(/Task tag creation failed/)).toBeNull();
    expect(guard?.isUnsafe()).toBe(false);
  });

  it("flushes a debounced draft on unmount", async () => {
    const putTaskAnnotation = vi.fn().mockResolvedValue(
      taskAnnotation("HumanEval/42", { note: "unmounted" }),
    );
    const api = fakeApi({
      getTaskAnnotation: vi.fn().mockResolvedValue(null),
      putTaskAnnotation,
    });
    const view = render(editor(api));
    fireEvent.change(await screen.findByLabelText("Task note"), {
      target: { value: "unmounted" },
    });

    view.unmount();

    await waitFor(() => expect(putTaskAnnotation).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ note: "unmounted" }),
    ));
  });
});
