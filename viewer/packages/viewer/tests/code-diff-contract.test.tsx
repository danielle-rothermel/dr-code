import { act, render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const diffMocks = vi.hoisted(() => ({
  diffView: vi.fn<(props: Record<string, unknown>) => null>(() => null),
  generateDiffFile: vi.fn<(...args: string[]) => unknown>(),
  getDiffViewHighlighter: vi.fn<() => Promise<unknown>>(),
}));

vi.mock("@git-diff-view/file", () => ({
  generateDiffFile: diffMocks.generateDiffFile,
}));

vi.mock("@git-diff-view/react", () => ({
  DiffModeEnum: {
    Split: "split-enum",
    Unified: "unified-enum",
  },
  DiffView: diffMocks.diffView,
}));

vi.mock("@git-diff-view/shiki", () => ({
  getDiffViewHighlighter: diffMocks.getDiffViewHighlighter,
}));

import { CodeDiff } from "../src/code-diff.js";

const OLD_CONTENT = "const answer = 1;";
const NEW_CONTENT = "const answer = 2;";
const HIGHLIGHTER = { name: "controlled-highlighter" };

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((onResolve) => {
    resolve = onResolve;
  });
  return { promise, resolve };
}

function diffFileMock() {
  return {
    initTheme: vi.fn(),
    initRaw: vi.fn(),
    initSyntax: vi.fn(),
    buildSplitDiffLines: vi.fn(),
    buildUnifiedDiffLines: vi.fn(),
  };
}

async function waitForDiffView(): Promise<Record<string, unknown>> {
  await waitFor(() => {
    expect(diffMocks.diffView).toHaveBeenCalledOnce();
  });
  const props = diffMocks.diffView.mock.calls[0]?.[0];
  if (props === undefined) throw new Error("DiffView was not called");
  return props;
}

describe("CodeDiff contract", () => {
  beforeEach(() => {
    diffMocks.diffView.mockReset();
    diffMocks.diffView.mockReturnValue(null);
    diffMocks.generateDiffFile.mockReset();
    diffMocks.getDiffViewHighlighter.mockReset();
  });

  it("maps the default unified presentation to the diff renderer", async () => {
    const file = diffFileMock();
    diffMocks.generateDiffFile.mockReturnValue(file);
    diffMocks.getDiffViewHighlighter.mockResolvedValue(HIGHLIGHTER);

    render(<CodeDiff oldContent={OLD_CONTENT} newContent={NEW_CONTENT} />);

    const props = await waitForDiffView();
    expect(diffMocks.generateDiffFile).toHaveBeenCalledWith(
      "before",
      OLD_CONTENT,
      "after",
      NEW_CONTENT,
      "python",
      "python",
    );
    expect(file.initTheme).toHaveBeenCalledWith("light");
    expect(file.initRaw).toHaveBeenCalledOnce();
    expect(file.initSyntax).toHaveBeenCalledWith({
      registerHighlighter: HIGHLIGHTER,
    });
    expect(file.buildSplitDiffLines).toHaveBeenCalledOnce();
    expect(file.buildUnifiedDiffLines).toHaveBeenCalledOnce();
    expect(props).toMatchObject({
      diffFile: file,
      diffViewMode: "unified-enum",
      diffViewTheme: "light",
      diffViewFontSize: 11,
      diffViewHighlight: true,
      diffViewWrap: true,
    });
  });

  it("maps split, dark, language, and custom names to vendor inputs", async () => {
    const file = diffFileMock();
    diffMocks.generateDiffFile.mockReturnValue(file);
    diffMocks.getDiffViewHighlighter.mockResolvedValue(HIGHLIGHTER);

    render(
      <CodeDiff
        oldContent={OLD_CONTENT}
        newContent={NEW_CONTENT}
        oldName="before.ts"
        newName="after.ts"
        lang="typescript"
        mode="split"
        theme="dark"
      />,
    );

    const props = await waitForDiffView();
    expect(diffMocks.generateDiffFile).toHaveBeenCalledWith(
      "before.ts",
      OLD_CONTENT,
      "after.ts",
      NEW_CONTENT,
      "typescript",
      "typescript",
    );
    expect(file.initTheme).toHaveBeenCalledWith("dark");
    expect(props).toMatchObject({
      diffFile: file,
      diffViewMode: "split-enum",
      diffViewTheme: "dark",
      diffViewFontSize: 10.5,
    });
  });

  it("only initializes the current diff after a deferred load", async () => {
    const pending = deferred<unknown>();
    const obsoleteFile = diffFileMock();
    const currentFile = diffFileMock();
    diffMocks.generateDiffFile
      .mockReturnValueOnce(obsoleteFile)
      .mockReturnValueOnce(currentFile);
    diffMocks.getDiffViewHighlighter.mockReturnValue(pending.promise);

    const { container, rerender } = render(
      <CodeDiff oldContent={OLD_CONTENT} newContent={NEW_CONTENT} />,
    );
    rerender(
      <CodeDiff
        oldContent="const answer: number = 2;"
        newContent="const answer: number = 3;"
        oldName="before.ts"
        newName="after.ts"
        lang="typescript"
        theme="dark"
      />,
    );

    expect(container.querySelector("code")?.textContent).toBe(
      "const answer: number = 3;",
    );

    await act(async () => {
      pending.resolve(HIGHLIGHTER);
      await pending.promise;
    });

    expect(obsoleteFile.initTheme).not.toHaveBeenCalled();
    expect(currentFile.initTheme).toHaveBeenCalledWith("dark");
    expect(currentFile.initSyntax).toHaveBeenCalledWith({
      registerHighlighter: HIGHLIGHTER,
    });
    expect(diffMocks.diffView).toHaveBeenCalledOnce();
  });

  it("does not initialize a diff after unmounting during a deferred load", async () => {
    const pending = deferred<unknown>();
    const file = diffFileMock();
    diffMocks.generateDiffFile.mockReturnValue(file);
    diffMocks.getDiffViewHighlighter.mockReturnValue(pending.promise);

    const { unmount } = render(
      <CodeDiff oldContent={OLD_CONTENT} newContent={NEW_CONTENT} />,
    );
    unmount();

    await act(async () => {
      pending.resolve(HIGHLIGHTER);
      await pending.promise;
    });

    expect(file.initTheme).not.toHaveBeenCalled();
    expect(diffMocks.diffView).not.toHaveBeenCalled();
  });
});
