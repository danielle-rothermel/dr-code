import { useEffect, useMemo, useState, type ReactNode } from "react";

import { StatusBadge } from "@dr-code/viewer";

import {
  FailureExamplesLoader,
  filterFailureEntries,
  paginateFailureEntries,
  type Example,
  type FailureBrowserGroup,
  type FailureBrowserSummary,
  type FailureGroupIndex,
} from "./data";

const numberFormatter = new Intl.NumberFormat("en-US");

type LoadState<T> =
  | { status: "idle" }
  | { key: string; status: "loading" }
  | { error: string; key: string; status: "error" }
  | { data: T; key: string; status: "success" };

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "An unknown error occurred";
}

function humanize(value: string): string {
  return value.replaceAll("_", " ");
}

function groupLabel(group: FailureBrowserGroup): string {
  return `${humanize(group.failure_code)} · ${humanize(group.failed_step)}`;
}

export function FailureBrowser({
  browser,
  loader,
  renderExample,
}: {
  browser: FailureBrowserSummary;
  loader: FailureExamplesLoader;
  renderExample: (example: Example) => ReactNode;
}) {
  const [selectedIndexPath, setSelectedIndexPath] = useState("");
  const [indexRetry, setIndexRetry] = useState(0);
  const [indexState, setIndexState] = useState<LoadState<FailureGroupIndex>>({ status: "idle" });
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [selectedId, setSelectedId] = useState("");
  const [detailRetry, setDetailRetry] = useState(0);
  const [detailState, setDetailState] = useState<LoadState<Example>>({ status: "idle" });

  const selectedGroup = browser.groups.find(({ index_path }) => index_path === selectedIndexPath);

  useEffect(() => {
    if (selectedGroup === undefined) return;
    const key = selectedGroup.index_path;
    let current = true;
    setIndexState({ key, status: "loading" });
    void loader.loadIndex(selectedGroup).then(
      (data) => {
        if (current) setIndexState({ data, key, status: "success" });
      },
      (error: unknown) => {
        if (current) setIndexState({ error: errorMessage(error), key, status: "error" });
      },
    );
    return () => {
      current = false;
    };
  }, [indexRetry, loader, selectedGroup]);

  const currentIndex =
    indexState.status === "success" && indexState.key === selectedGroup?.index_path
      ? indexState.data
      : undefined;

  const filteredEntries = useMemo(
    () => filterFailureEntries(currentIndex?.entries ?? [], search),
    [currentIndex, search],
  );
  const failurePage = useMemo(
    () => paginateFailureEntries(filteredEntries, page),
    [filteredEntries, page],
  );
  const selectedEntry =
    failurePage.entries.find(({ sample_id }) => sample_id === selectedId) ??
    failurePage.entries[0];
  const detailKey =
    selectedEntry === undefined || selectedGroup === undefined
      ? undefined
      : `${selectedGroup.index_path}\0${selectedEntry.sample_id}`;

  useEffect(() => {
    if (selectedEntry === undefined || selectedGroup === undefined || detailKey === undefined) return;
    let current = true;
    setDetailState({ key: detailKey, status: "loading" });
    void loader.loadDetail(selectedEntry, selectedGroup.failure_code).then(
      (data) => {
        if (current) setDetailState({ data, key: detailKey, status: "success" });
      },
      (error: unknown) => {
        if (current) setDetailState({ error: errorMessage(error), key: detailKey, status: "error" });
      },
    );
    return () => {
      current = false;
    };
  }, [detailKey, detailRetry, loader, selectedEntry, selectedGroup]);

  function selectGroup(group: FailureBrowserGroup) {
    setSelectedIndexPath(group.index_path);
    setIndexState({ key: group.index_path, status: "loading" });
    setDetailState({ status: "idle" });
    setSearch("");
    setPage(1);
    setSelectedId("");
  }

  function updateSearch(value: string) {
    setSearch(value);
    setPage(1);
    setSelectedId("");
  }

  if (browser.groups.length === 0) {
    return (
      <section className="failures" id="failures" aria-labelledby="failures-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">failure explorer</p>
            <h2 id="failures-title">All terminal preprocessing failures</h2>
          </div>
        </div>
        <p className="empty-copy">This snapshot has no terminal failure examples.</p>
      </section>
    );
  }

  return (
    <section className="failures" id="failures" aria-labelledby="failures-title">
      <div className="section-heading failure-heading">
        <div>
          <p className="eyebrow">failure explorer</p>
          <h2 id="failures-title">Inspect every nonblank response that produced no final candidate.</h2>
          <p>
            {numberFormatter.format(browser.total_count)} examples grouped by their terminal failure cause.
            Indexes and response details load from packaged static shards as you explore.
          </p>
        </div>
      </div>

      <div className="failure-groups" aria-label="Terminal failure causes">
        {browser.groups.map((group) => (
          <button
            aria-pressed={group.index_path === selectedGroup?.index_path}
            className={
              group.index_path === selectedGroup?.index_path
                ? "failure-group failure-group--selected"
                : "failure-group"
            }
            key={group.index_path}
            onClick={() => selectGroup(group)}
            type="button"
          >
            <span>{groupLabel(group)}</span>
            <strong>{numberFormatter.format(group.count)}</strong>
            <small>examples</small>
          </button>
        ))}
      </div>

      {selectedGroup === undefined ? (
        <div className="panel failure-panel failure-prompt">
          <p>Select a terminal cause to load its failure index.</p>
        </div>
      ) : <div className="panel failure-panel">
        <div className="section-heading compact">
          <div>
            <p className="eyebrow">active cause</p>
            <h2>{selectedGroup ? groupLabel(selectedGroup) : "Failure examples"}</h2>
          </div>
          {currentIndex !== undefined && (
            <label className="failure-search">
              Search failures
              <input
                onChange={(event) => updateSearch(event.target.value)}
                placeholder="sample ID, outcome, context, rejection…"
                value={search}
              />
            </label>
          )}
        </div>

        {indexState.status === "loading" && indexState.key === selectedGroup.index_path && (
          <p className="failure-state" role="status">Loading failure index…</p>
        )}
        {indexState.status === "error" && indexState.key === selectedGroup.index_path && (
          <div className="failure-state failure-state--error" role="alert">
            <strong>Could not load this failure group.</strong>
            <span>{indexState.error}</span>
            <button onClick={() => setIndexRetry((value) => value + 1)} type="button">
              Retry failure group
            </button>
          </div>
        )}
        {currentIndex !== undefined && (
          <>
            <div className="failure-range">
              <span>
                Showing {numberFormatter.format(failurePage.start)}–{numberFormatter.format(failurePage.end)} of {numberFormatter.format(failurePage.total)}
              </span>
              {search !== "" && <small>{numberFormatter.format(currentIndex.count)} in this cause</small>}
            </div>

            {failurePage.entries.length === 0 ? (
              <p className="empty-copy">No failure matches that search.</p>
            ) : (
              <div className="failure-layout">
                <div>
                  <div className="example-list failure-list" aria-label="Filtered failures">
                    {failurePage.entries.map((entry) => (
                      <button
                        className={
                          selectedEntry?.sample_id === entry.sample_id
                            ? "example-card failure-card example-card--selected"
                            : "example-card failure-card"
                        }
                        key={entry.sample_id}
                        onClick={() => setSelectedId(entry.sample_id)}
                        type="button"
                      >
                        <StatusBadge status="failure">{humanize(entry.outcome)}</StatusBadge>
                        <strong>{entry.sample_id}</strong>
                        <span>
                          {entry.rejection_reasons.map(humanize).join(" · ") || humanize(entry.failed_step)}
                        </span>
                        <small>{numberFormatter.format(entry.raw_character_count)} response characters</small>
                      </button>
                    ))}
                  </div>
                  {failurePage.pageCount > 1 && (
                    <div className="failure-pagination" aria-label="Failure pages">
                      <button
                        disabled={failurePage.page === 1}
                        onClick={() => {
                          setPage((value) => value - 1);
                          setSelectedId("");
                        }}
                        type="button"
                      >
                        Previous
                      </button>
                      <span>Page {failurePage.page} of {failurePage.pageCount}</span>
                      <button
                        disabled={failurePage.page === failurePage.pageCount}
                        onClick={() => {
                          setPage((value) => value + 1);
                          setSelectedId("");
                        }}
                        type="button"
                      >
                        Next
                      </button>
                    </div>
                  )}
                </div>

                {detailState.status === "loading" && detailState.key === detailKey && (
                  <p className="failure-state" role="status">Loading failure details…</p>
                )}
                {detailState.status === "error" && detailState.key === detailKey && (
                  <div className="failure-state failure-state--error" role="alert">
                    <strong>Could not load this example.</strong>
                    <span>{detailState.error}</span>
                    <button onClick={() => setDetailRetry((value) => value + 1)} type="button">
                      Retry failure detail
                    </button>
                  </div>
                )}
                {detailState.status === "success" && detailState.key === detailKey && (
                  renderExample(detailState.data)
                )}
              </div>
            )}
          </>
        )}
      </div>}
    </section>
  );
}
