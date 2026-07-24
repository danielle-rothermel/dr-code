import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@dr-code/viewer", () => ({
  CodeBlock: ({ code }: { code: string }) => <pre>{code}</pre>,
  StatusBadge: ({ children }: { children: ReactNode }) => <span>{children}</span>,
}));

import type { Annotation, ExampleDetail, ReviewExamplesQuery } from "../src/api";
import {
  ANNOTATION_NOTE_MAX_LENGTH,
  ANNOTATION_TAG_IDS_MAX_COUNT,
  TAG_NAME_MAX_LENGTH,
  TAG_NAME_WHITESPACE_CODE_POINTS,
  contractLength,
  isAnnotationNoteInContract,
  isTagNameInContract,
  normalizeTagName,
} from "../src/annotation-contract";
import { Review } from "../src/review";
import { detail, fakeApi } from "./fixtures";

afterEach(cleanup);

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, reject, resolve };
}

function example(sampleId: string, overrides: Partial<ExampleDetail> = {}): ExampleDetail {
  return {
    ...detail,
    decoder_output_sha256: `output-${sampleId}`,
    sample_id: sampleId,
    ...overrides,
  };
}

function getCard(sampleId: string) {
  return screen.getByRole("article", { name: `Example ${sampleId}` });
}

function findCard(sampleId: string) {
  return screen.findByRole("article", { name: `Example ${sampleId}` });
}

describe("Review", () => {
  it("uses the pinned Unicode scalar and tag whitespace contract", () => {
    expect(TAG_NAME_WHITESPACE_CODE_POINTS).toEqual([
      0x0009, 0x000a, 0x000b, 0x000c, 0x000d, 0x0020, 0x0085, 0x00a0,
      0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006,
      0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000,
    ]);
    const whitespace = TAG_NAME_WHITESPACE_CODE_POINTS
      .map((codePoint) => String.fromCodePoint(codePoint)).join("");
    expect(normalizeTagName(`left${whitespace}right`)).toBe("left right");
    for (const codePoint of [0x001c, 0x001d, 0x001e, 0x001f, 0xfeff]) {
      const separator = String.fromCodePoint(codePoint);
      expect(normalizeTagName(`left${separator}right`)).toBe(`left${separator}right`);
    }

    expect(contractLength("😀")).toBe(1);
    expect(isTagNameInContract("😀".repeat(TAG_NAME_MAX_LENGTH))).toBe(true);
    expect(isAnnotationNoteInContract("😀".repeat(ANNOTATION_NOTE_MAX_LENGTH))).toBe(true);
    for (const surrogate of ["\ud800", "\udfff"]) {
      expect(isTagNameInContract(surrogate)).toBe(false);
      expect(isAnnotationNoteInContract(surrogate)).toBe(false);
    }
  });

  it("requests complete page items, renders every card, and uses semantic review layout classes", async () => {
    const items = [
      example("sample-1", {
        context: {
          content_sha256: "content-sha",
          has_prompt: true,
          source_record_id: "source-record-1",
          warnings: "truncated source",
        },
      }),
      example("sample-2"),
    ];
    const api = fakeApi({
      getReviewExamples: vi.fn().mockResolvedValue({ items, limit: 10, offset: 0, total: 2 }),
    });

    const { container } = render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={[]} />);

    expect(await findCard("sample-1")).toBeTruthy();
    expect(getCard("sample-2")).toBeTruthy();
    expect(api.getReviewExamples).toHaveBeenCalledWith("baseline", {
      cause: "syntax error",
      failed_step: "compile",
      failure_code: "syntax_error",
      limit: 10,
      offset: 0,
      search: "",
    });
    expect(container.querySelectorAll(".review-example-card")).toHaveLength(2);
    expect(container.querySelector(".example-browser")).toBeNull();
    expect(container.querySelector(".example-list")).toBeNull();
    expect(screen.queryByText("unreviewed")).toBeNull();
    expect(container.querySelector(".review-example-card")?.className).toContain("review-example-card--three-one");
    expect(container.querySelector(".review-example-main")).toBeTruthy();
    expect(container.querySelector(".annotation-rail")).toBeTruthy();

    const metadata = (label: string) => screen.getByText(label).closest("div");
    expect(metadata("source record id")?.className).toContain("metadata-field--half");
    expect(metadata("content sha256")?.className).toContain("metadata-field--half");
    expect(metadata("warnings")?.className).toContain("metadata-field--full");
    expect(metadata("has prompt")?.className).toContain("metadata-field--compact");

    const firstCard = getCard("sample-1");
    const decoder = within(firstCard).getByRole("region", { name: "Decoder output for sample-1" });
    expect(decoder.previousElementSibling?.className).toBe("failure-reason");
    const details = firstCard.querySelector("details.review-details") as HTMLDetailsElement;
    expect(details.open).toBe(false);
    expect(decoder.nextElementSibling).toBe(details);
    expect(within(details).getByText("sample-1")).toBeTruthy();
    expect(firstCard.querySelector(".failure-reason h3")).toBeNull();
  });

  it("provides page selectors, page buttons, page sizes, and resets page on each filter boundary", async () => {
    const getReviewExamples = vi.fn(async (_runId: string, query: ReviewExamplesQuery) => ({
      items: [example(`sample-${query.offset + 1}`)],
      limit: query.limit,
      offset: query.offset,
      total: 60,
    }));
    const api = fakeApi({ getReviewExamples });
    const view = render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={[]} />);

    await findCard("sample-1");
    expect((screen.getByLabelText("Page size") as HTMLSelectElement).value).toBe("10");
    expect((screen.getByLabelText("Page number") as HTMLSelectElement).selectedOptions[0]?.textContent).toBe("Page 1 of 6");
    expect((screen.getByRole("button", { name: "Previous page" }) as HTMLButtonElement).disabled).toBe(true);

    fireEvent.change(screen.getByLabelText("Page number"), { target: { value: "3" } });
    await waitFor(() => expect(getReviewExamples).toHaveBeenCalledWith("baseline", expect.objectContaining({ offset: 20 })));
    expect(await findCard("sample-21")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Previous page" }));
    await waitFor(() => expect(getReviewExamples).toHaveBeenCalledWith("baseline", expect.objectContaining({ limit: 10, offset: 10 })));
    expect(await findCard("sample-11")).toBeTruthy();

    const callsBeforeTyping = getReviewExamples.mock.calls.length;
    fireEvent.change(screen.getByLabelText("Search"), { target: { value: "n" } });
    fireEvent.change(screen.getByLabelText("Search"), { target: { value: "needle" } });
    expect(getReviewExamples).toHaveBeenCalledTimes(callsBeforeTyping);
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() => expect(getReviewExamples).toHaveBeenCalledWith("baseline", expect.objectContaining({ offset: 0, search: "needle" })));
    expect(getReviewExamples).toHaveBeenCalledTimes(callsBeforeTyping + 1);

    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    await waitFor(() => expect(getReviewExamples).toHaveBeenCalledWith("baseline", expect.objectContaining({ offset: 10, search: "needle" })));
    fireEvent.change(screen.getByLabelText("Page size"), { target: { value: "25" } });
    await waitFor(() => expect(getReviewExamples).toHaveBeenCalledWith("baseline", expect.objectContaining({ limit: 25, offset: 0 })));
    expect((screen.getByLabelText("Page number") as HTMLSelectElement).selectedOptions[0]?.textContent).toBe("Page 1 of 3");

    const nullCauseOption = screen.getByRole("option", { name: /Literal response/ }) as HTMLOptionElement;
    fireEvent.change(screen.getByLabelText("Failure group"), { target: { value: nullCauseOption.value } });
    await waitFor(() => expect(getReviewExamples).toHaveBeenCalledWith("baseline", expect.objectContaining({
      cause_is_null: true,
      offset: 0,
    })));

    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    await waitFor(() => expect(getReviewExamples).toHaveBeenCalledWith("baseline", expect.objectContaining({ offset: 25 })));
    view.rerender(<Review api={api} onTagCreated={vi.fn()} runId="candidate" tags={[]} />);
    await waitFor(() => expect(getReviewExamples).toHaveBeenCalledWith("candidate", expect.objectContaining({ offset: 0 })));
  });

  it("maps verdict labels and saves comments and tags while the verdict is null", async () => {
    const tag = { name: "markdown fence", tag_id: "tag-1" };
    const api = fakeApi({
      putAnnotation: vi.fn(async (_identity, input) => ({ note: input.note, tags: input.tag_ids.includes(tag.tag_id) ? [tag] : [], verdict: input.verdict })),
    });
    render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={[tag]} />);

    const card = await findCard("sample-1");
    const radios = within(card).getAllByRole("radio");
    expect(radios.map((radio) => radio.parentElement?.textContent)).toEqual(["Unlabeled", "Flag", "Verify"]);
    expect((within(card).getByRole("radio", { name: "Unlabeled" }) as HTMLInputElement).checked).toBe(true);

    fireEvent.change(within(card).getByLabelText("Comment"), { target: { value: "keep while unlabeled" } });
    fireEvent.blur(within(card).getByLabelText("Comment"));
    await waitFor(() => expect(api.putAnnotation).toHaveBeenLastCalledWith(expect.anything(), {
      note: "keep while unlabeled",
      tag_ids: [],
      verdict: null,
    }));

    fireEvent.click(within(card).getByRole("checkbox", { name: "markdown fence" }));
    await waitFor(() => expect(api.putAnnotation).toHaveBeenLastCalledWith(expect.anything(), {
      note: "keep while unlabeled",
      tag_ids: ["tag-1"],
      verdict: null,
    }));

    fireEvent.click(within(card).getByRole("radio", { name: "Flag" }));
    await waitFor(() => expect(api.putAnnotation).toHaveBeenLastCalledWith(expect.anything(), expect.objectContaining({ verdict: "should_be_parseable" })));
    fireEvent.click(within(card).getByRole("radio", { name: "Verify" }));
    await waitFor(() => expect(api.putAnnotation).toHaveBeenLastCalledWith(expect.anything(), expect.objectContaining({ verdict: "expected_no_code" })));
    fireEvent.click(within(card).getByRole("radio", { name: "Unlabeled" }));
    await waitFor(() => expect(api.putAnnotation).toHaveBeenLastCalledWith(expect.anything(), expect.objectContaining({ verdict: null })));
  });

  it("keeps note state and saves within the exact annotation note maximum", async () => {
    const putAnnotation = vi.fn(async (_identity, input) => ({
      note: input.note,
      tags: [],
      verdict: input.verdict,
    }));
    const api = fakeApi({ putAnnotation });
    render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={[]} />);

    const note = await screen.findByLabelText("Comment") as HTMLTextAreaElement;
    const exact = "n".repeat(ANNOTATION_NOTE_MAX_LENGTH);
    fireEvent.change(note, { target: { value: exact } });
    fireEvent.blur(note);
    await waitFor(() => expect(putAnnotation).toHaveBeenLastCalledWith(expect.anything(), {
      note: exact,
      tag_ids: [],
      verdict: null,
    }));
    expect(note.maxLength).toBe(ANNOTATION_NOTE_MAX_LENGTH * 2);

    const callsAtMaximum = putAnnotation.mock.calls.length;
    fireEvent.change(note, { target: { value: `${exact}x` } });
    fireEvent.blur(note);
    expect(note.value).toBe(exact);
    expect(putAnnotation).toHaveBeenCalledTimes(callsAtMaximum);

    const exactUnicode = "😀".repeat(ANNOTATION_NOTE_MAX_LENGTH);
    fireEvent.change(note, { target: { value: exactUnicode } });
    fireEvent.blur(note);
    await waitFor(() => expect(putAnnotation).toHaveBeenLastCalledWith(expect.anything(), {
      note: exactUnicode,
      tag_ids: [],
      verdict: null,
    }));
    fireEvent.change(note, { target: { value: `${exactUnicode}😀` } });
    expect(note.value).toBe(exactUnicode);
    fireEvent.change(note, { target: { value: "\ud800" } });
    expect(note.value).toBe(exactUnicode);
  });

  it("prevents a 101st distinct tag while allowing replacement at the cap", async () => {
    const tags = Array.from({ length: ANNOTATION_TAG_IDS_MAX_COUNT + 1 }, (_value, index) => ({
      name: `tag ${index.toString().padStart(3, "0")}`,
      tag_id: `tag-${index.toString().padStart(3, "0")}`,
    }));
    const selected = tags.slice(0, ANNOTATION_TAG_IDS_MAX_COUNT);
    const item = example("sample-1", {
      annotation: { note: null, tags: selected, verdict: null },
    });
    const putAnnotation = vi.fn(async (_identity, input) => ({
      note: input.note,
      tags: tags.filter(({ tag_id }) => input.tag_ids.includes(tag_id)),
      verdict: input.verdict,
    }));
    const api = fakeApi({
      getReviewExamples: vi.fn().mockResolvedValue({ items: [item], limit: 10, offset: 0, total: 1 }),
      putAnnotation,
    });
    render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={tags} />);

    const card = await findCard("sample-1");
    const overflow = within(card).getByRole("checkbox", { name: tags.at(-1)?.name }) as HTMLInputElement;
    expect(overflow.disabled).toBe(true);
    expect((within(card).getByRole("button", { name: "Create and select" }) as HTMLButtonElement).disabled).toBe(true);

    fireEvent.click(within(card).getByRole("checkbox", { name: tags[0]?.name }));
    await waitFor(() => expect(putAnnotation).toHaveBeenCalled());
    expect(overflow.disabled).toBe(false);
    fireEvent.click(overflow);
    await waitFor(() => expect(putAnnotation).toHaveBeenLastCalledWith(expect.anything(), expect.objectContaining({
      tag_ids: expect.arrayContaining([tags.at(-1)?.tag_id]),
    })));
    const lastInput = putAnnotation.mock.calls.at(-1)?.[1];
    expect(new Set(lastInput?.tag_ids).size).toBe(ANNOTATION_TAG_IDS_MAX_COUNT);
  });

  it("creates only tag names within the normalized maximum", async () => {
    const createTag = vi.fn(async (name: string) => ({ name, tag_id: "created" }));
    const api = fakeApi({ createTag });
    render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={[]} />);

    const input = await screen.findByLabelText("Create tag") as HTMLInputElement;
    const button = screen.getByRole("button", { name: "Create and select" }) as HTMLButtonElement;
    const exact = "x".repeat(TAG_NAME_MAX_LENGTH);
    fireEvent.change(input, { target: { value: exact } });
    expect(button.disabled).toBe(false);
    fireEvent.click(button);
    await waitFor(() => expect(createTag).toHaveBeenCalledWith(exact));

    await waitFor(() => expect(input.value).toBe(""));
    fireEvent.change(input, { target: { value: "x".repeat(TAG_NAME_MAX_LENGTH + 1) } });
    expect(button.disabled).toBe(true);
    fireEvent.click(button);
    expect(createTag).toHaveBeenCalledTimes(1);

    fireEvent.change(input, { target: { value: `left${" ".repeat(200)}right` } });
    expect(button.disabled).toBe(false);
    fireEvent.click(button);
    await waitFor(() => expect(createTag).toHaveBeenLastCalledWith("left right"));

    await waitFor(() => expect(input.value).toBe(""));
    const exactUnicode = "😀".repeat(TAG_NAME_MAX_LENGTH);
    fireEvent.change(input, { target: { value: exactUnicode } });
    expect(button.disabled).toBe(false);
    fireEvent.change(input, { target: { value: `${exactUnicode}😀` } });
    expect(button.disabled).toBe(true);
    fireEvent.change(input, { target: { value: "\ud800" } });
    expect(button.disabled).toBe(true);
  });

  it("creates a shared tag, selects it for an unlabeled card, and persists it", async () => {
    const created = { name: "new tag", tag_id: "created-tag" };
    const onTagCreated = vi.fn();
    const api = fakeApi({ createTag: vi.fn().mockResolvedValue(created) });
    render(<Review api={api} onTagCreated={onTagCreated} runId="baseline" tags={[]} />);

    const card = await findCard("sample-1");
    fireEvent.change(within(card).getByLabelText("Create tag"), { target: { value: "new tag" } });
    fireEvent.click(within(card).getByRole("button", { name: "Create and select" }));

    await waitFor(() => expect(onTagCreated).toHaveBeenCalledWith(created));
    await waitFor(() => expect(api.putAnnotation).toHaveBeenCalledWith(expect.anything(), {
      note: "",
      tag_ids: ["created-tag"],
      verdict: null,
    }));
  });

  it("keeps failed tag creation protected through unrelated saves and retries it through navigation", async () => {
    const created = { name: "retry tag", tag_id: "retry-tag" };
    const createTag = vi.fn()
      .mockRejectedValueOnce(new Error("tag database is locked"))
      .mockRejectedValueOnce(new Error("tag database is still locked"))
      .mockResolvedValue(created);
    const putAnnotation = vi.fn(async (_identity, input) => ({
      note: input.note,
      tags: input.tag_ids.includes(created.tag_id) ? [created] : [],
      verdict: input.verdict,
    }));
    const getReviewExamples = vi.fn(async (_runId: string, query: ReviewExamplesQuery) => ({
      items: [query.offset === 0 ? example("sample-1") : example("sample-11")],
      limit: query.limit,
      offset: query.offset,
      total: 11,
    }));
    const onTagCreated = vi.fn();
    const api = fakeApi({ createTag, getReviewExamples, putAnnotation });
    render(<Review api={api} onTagCreated={onTagCreated} runId="baseline" tags={[]} />);

    const card = await findCard("sample-1");
    const tagInput = within(card).getByLabelText("Create tag") as HTMLInputElement;
    fireEvent.change(tagInput, { target: { value: "retry tag" } });
    fireEvent.click(within(card).getByRole("button", { name: "Create and select" }));

    expect(await within(card).findByText("Tag save failed")).toBeTruthy();
    expect(within(card).getByRole("alert").textContent).toContain("tag database is locked");
    expect(tagInput.value).toBe("retry tag");
    expect(within(card).getByRole("button", { name: "Retry create and select" })).toBeTruthy();

    fireEvent.click(within(card).getByRole("radio", { name: "Flag" }));
    await waitFor(() => expect(putAnnotation).toHaveBeenCalledWith(expect.anything(), expect.objectContaining({
      tag_ids: [],
      verdict: "should_be_parseable",
    })));
    expect(within(card).getByText("Tag save failed")).toBeTruthy();
    expect(within(card).getByRole("alert").textContent).toContain("tag database is locked");
    const unsafeEvent = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(unsafeEvent);
    expect(unsafeEvent.defaultPrevented).toBe(true);

    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    expect(await screen.findByText("Navigation blocked")).toBeTruthy();
    expect(getCard("sample-1")).toBeTruthy();
    expect(createTag).toHaveBeenCalledTimes(2);
    expect(tagInput.value).toBe("retry tag");

    fireEvent.click(screen.getByRole("button", { name: "Retry pending saves" }));
    await waitFor(() => expect(createTag).toHaveBeenCalledTimes(3));
    await waitFor(() => expect(putAnnotation).toHaveBeenCalledWith(expect.anything(), {
      note: "",
      tag_ids: ["retry-tag"],
      verdict: "should_be_parseable",
    }));
    await waitFor(() => expect(screen.queryByText("Navigation blocked")).toBeNull());
    expect(onTagCreated).toHaveBeenCalledWith(created);
    expect(tagInput.value).toBe("");

    const cleanEvent = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(cleanEvent);
    expect(cleanEvent.defaultPrevented).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    await findCard("sample-11");
  });

  it("applies a submitted search after guarded saves without racing later draft text", async () => {
    const pendingSave = deferred<Annotation>();
    const getReviewExamples = vi.fn(async (_runId: string, query: ReviewExamplesQuery) => ({
      items: [example(query.search === "applied" ? "applied-result" : "sample-1")],
      limit: query.limit,
      offset: query.offset,
      total: 1,
    }));
    const api = fakeApi({ getReviewExamples, putAnnotation: vi.fn().mockReturnValue(pendingSave.promise) });
    render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={[]} />);

    fireEvent.click(await screen.findByRole("radio", { name: "Flag" }));
    fireEvent.change(screen.getByLabelText("Search"), { target: { value: "applied" } });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    fireEvent.change(screen.getByLabelText("Search"), { target: { value: "still drafting" } });
    expect(getReviewExamples).not.toHaveBeenCalledWith("baseline", expect.objectContaining({ search: "applied" }));
    expect(getReviewExamples).not.toHaveBeenCalledWith("baseline", expect.objectContaining({ search: "still drafting" }));

    await act(async () => {
      pendingSave.resolve({ note: "", tags: [], verdict: "should_be_parseable" });
      await pendingSave.promise;
    });
    await findCard("applied-result");
    expect(getReviewExamples).toHaveBeenCalledWith("baseline", expect.objectContaining({ search: "applied" }));
    expect(getReviewExamples).not.toHaveBeenCalledWith("baseline", expect.objectContaining({ search: "still drafting" }));
    expect((screen.getByLabelText("Search") as HTMLInputElement).value).toBe("still drafting");
  });

  it("refetches saved annotation state when navigating away from and back to a page", async () => {
    const annotations = new Map<string, Annotation>();
    const first = example("sample-1");
    const second = example("sample-11");
    const api = fakeApi({
      getReviewExamples: vi.fn(async (_runId: string, query: ReviewExamplesQuery) => ({
        items: [(query.offset === 0 ? first : second)].map((item) => ({ ...item, annotation: annotations.get(item.sample_id) ?? null })),
        limit: query.limit,
        offset: query.offset,
        total: 11,
      })),
      putAnnotation: vi.fn(async (identity, input) => {
        const annotation = { note: input.note, tags: [], verdict: input.verdict };
        annotations.set(identity.sample_id, annotation);
        return annotation;
      }),
    });
    render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={[]} />);

    fireEvent.click(await screen.findByRole("radio", { name: "Flag" }));
    await screen.findByText("Saved");
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    await findCard("sample-11");
    fireEvent.click(screen.getByRole("button", { name: "Previous page" }));
    await findCard("sample-1");
    expect((screen.getByRole("radio", { name: "Flag" }) as HTMLInputElement).checked).toBe(true);
  });

  it("saves multiple cards independently and concurrently", async () => {
    const firstSave = deferred<Annotation>();
    const secondSave = deferred<Annotation>();
    const putAnnotation = vi.fn()
      .mockReturnValueOnce(firstSave.promise)
      .mockReturnValueOnce(secondSave.promise);
    const api = fakeApi({
      getReviewExamples: vi.fn().mockResolvedValue({ items: [example("sample-1"), example("sample-2")], limit: 10, offset: 0, total: 2 }),
      putAnnotation,
    });
    render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={[]} />);

    const firstCard = await findCard("sample-1");
    const secondCard = getCard("sample-2");
    fireEvent.click(within(firstCard).getByRole("radio", { name: "Flag" }));
    fireEvent.click(within(secondCard).getByRole("radio", { name: "Verify" }));
    expect(putAnnotation).toHaveBeenCalledTimes(2);
    expect(putAnnotation.mock.calls.map(([identity]) => identity.sample_id)).toEqual(["sample-1", "sample-2"]);

    await act(async () => {
      firstSave.resolve({ note: "", tags: [], verdict: "should_be_parseable" });
      secondSave.resolve({ note: "", tags: [], verdict: "expected_no_code" });
      await Promise.all([firstSave.promise, secondSave.promise]);
    });
    expect(screen.getAllByText("Saved")).toHaveLength(2);
  });

  it("admits only one rapid navigation while a save succeeds", async () => {
    const pending = deferred<Annotation>();
    const getReviewExamples = vi.fn(async (_runId: string, query: ReviewExamplesQuery) => ({
      items: [example(query.offset === 0 ? "sample-1" : "sample-11")],
      limit: query.limit,
      offset: query.offset,
      total: 11,
    }));
    const api = fakeApi({
      getReviewExamples,
      putAnnotation: vi.fn().mockReturnValue(pending.promise),
    });
    render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={[]} />);

    fireEvent.click(await screen.findByRole("radio", { name: "Flag" }));
    const next = screen.getByRole("button", { name: "Next page" });
    fireEvent.click(next);
    fireEvent.click(next);
    expect(getReviewExamples.mock.calls.filter(([, query]) => query.offset === 10)).toHaveLength(0);

    await act(async () => {
      pending.resolve({ note: "", tags: [], verdict: "should_be_parseable" });
      await pending.promise;
    });
    await findCard("sample-11");
    expect(getReviewExamples.mock.calls.filter(([, query]) => query.offset === 10)).toHaveLength(1);
  });

  it("clears the single-flight guard after failure so one retry can navigate", async () => {
    const failed = deferred<Annotation>();
    const getReviewExamples = vi.fn(async (_runId: string, query: ReviewExamplesQuery) => ({
      items: [example(query.offset === 0 ? "sample-1" : "sample-11")],
      limit: query.limit,
      offset: query.offset,
      total: 11,
    }));
    const putAnnotation = vi.fn()
      .mockReturnValueOnce(failed.promise)
      .mockResolvedValue({ note: "draft", tags: [], verdict: null });
    const api = fakeApi({ getReviewExamples, putAnnotation });
    render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={[]} />);

    fireEvent.change(await screen.findByLabelText("Comment"), { target: { value: "draft" } });
    const next = screen.getByRole("button", { name: "Next page" });
    fireEvent.click(next);
    fireEvent.click(next);
    await act(async () => {
      failed.reject(new Error("database is locked"));
      await expect(failed.promise).rejects.toThrow("database is locked");
    });
    expect(await screen.findByText("Navigation blocked")).toBeTruthy();
    expect(getReviewExamples.mock.calls.filter(([, query]) => query.offset === 10)).toHaveLength(0);

    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    await findCard("sample-11");
    expect(putAnnotation).toHaveBeenCalledTimes(2);
    expect(getReviewExamples.mock.calls.filter(([, query]) => query.offset === 10)).toHaveLength(1);
  });

  it("composes every card guard, blocks page navigation on one failure, and can recover", async () => {
    const first = example("sample-1", { annotation: { note: null, tags: [], verdict: null } });
    const failing = example("sample-2", { annotation: { note: null, tags: [], verdict: null } });
    const secondPage = example("sample-11");
    let hasFailed = false;
    const putAnnotation = vi.fn(async (identity, input) => {
      if (identity.sample_id === "sample-2" && !hasFailed) {
        hasFailed = true;
        throw new Error("database is locked");
      }
      return { note: input.note, tags: [], verdict: input.verdict };
    });
    const getReviewExamples = vi.fn(async (_runId: string, query: ReviewExamplesQuery) => ({
      items: query.offset === 0 ? [first, failing] : [secondPage],
      limit: query.limit,
      offset: query.offset,
      total: 11,
    }));
    const api = fakeApi({ getReviewExamples, putAnnotation });
    render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={[]} />);

    const firstCard = await findCard("sample-1");
    const failingCard = getCard("sample-2");
    fireEvent.change(within(firstCard).getByLabelText("Comment"), { target: { value: "first draft" } });
    fireEvent.change(within(failingCard).getByLabelText("Comment"), { target: { value: "recoverable draft" } });
    fireEvent.click(screen.getByRole("button", { name: "Next page" }));

    expect(await screen.findByText("Navigation blocked")).toBeTruthy();
    expect(getCard("sample-1")).toBeTruthy();
    expect((within(firstCard).getByLabelText("Comment") as HTMLTextAreaElement).value).toBe("first draft");
    expect((within(failingCard).getByLabelText("Comment") as HTMLTextAreaElement).value).toBe("recoverable draft");
    expect(putAnnotation.mock.calls.map(([identity]) => identity.sample_id).sort()).toEqual(["sample-1", "sample-2"]);
    expect(getReviewExamples).not.toHaveBeenCalledWith("baseline", expect.objectContaining({ offset: 10 }));

    fireEvent.click(screen.getByRole("button", { name: "Next page" }));
    await findCard("sample-11");
    expect(putAnnotation).toHaveBeenCalledTimes(3);
  });

  it("guards beforeunload while any card is saving", async () => {
    const pending = deferred<Annotation>();
    const api = fakeApi({ putAnnotation: vi.fn().mockReturnValue(pending.promise) });
    render(<Review api={api} onTagCreated={vi.fn()} runId="baseline" tags={[]} />);

    fireEvent.click(await screen.findByRole("radio", { name: "Flag" }));
    const unsafeEvent = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(unsafeEvent);
    expect(unsafeEvent.defaultPrevented).toBe(true);

    await act(async () => {
      pending.resolve({ note: "", tags: [], verdict: "should_be_parseable" });
      await pending.promise;
    });
    const cleanEvent = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(cleanEvent);
    expect(cleanEvent.defaultPrevented).toBe(false);
  });
});
