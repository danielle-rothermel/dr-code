# Local preprocessing waterfall and annotation viewer

## Goal

Replace the checked, static preprocessing report with a local single-user
application that can inspect any complete preprocessing run, compare compatible
runs, and persist human review of failed extractions. The viewer should make
every aggregate count traceable to the underlying examples and should query the
authoritative Parquet artifacts instead of copying them into frontend JSON.

The application is a development tool. It binds only to loopback, runs as one
process, and does not need authentication, multi-user coordination, a hosted
database, or deployment infrastructure.

## Product surface

The viewer has three focused surfaces:

1. **Waterfall** shows one run's progression from corpus rows through decoder
   output presence, nonblank output, extracted candidates, compilable candidates,
   top-level functions, and (when evaluation artifacts are supplied) tested and
   passing candidates. Selecting a stage or failure count opens its examples.
2. **Compare** places a baseline and candidate run on the same waterfall,
   reports count and rate deltas, and shows the transition matrix between terminal
   outcomes. Every transition is inspectable as an exact example list.
3. **Review** groups nonblank decoder outputs with no final function candidate by
   terminal failure cause. A reviewer can mark each example `should_be_parseable`
   or `expected_no_code`, clear that decision, add an optional note, select
   existing tags, and create new tags.

Comparison is available only when the two manifests identify the same corpus and
compatible semantic coordinates. The UI must show why an incompatible comparison
was rejected instead of presenting misleading deltas.

## Runtime architecture

```text
immutable run artifacts                 local mutable state
  corpus.parquet                          annotations
  preprocessing manifest                 tags
  results.parquet                         annotation_tags
  candidates.parquet                           |
  step_facts.parquet                           |
  rejections.parquet                           v
  optional evaluation artifacts -----> DuckDB query/service layer
                                                |
                                                v
                                            FastAPI
                                                |
                                                v
                                      React/Vite application
```

DuckDB has two deliberately separate responsibilities:

- Query the immutable Parquet relations in place. Run rows are not imported or
  duplicated in the database.
- Store the small local catalog and annotation state in one ignored `.duckdb`
  file.

FastAPI owns input validation and typed HTTP contracts. It serves both the API
and the built frontend from one loopback-only Uvicorn process. There is no raw
SQL API, no browser-supplied arbitrary artifact path, no CORS configuration, and
no multi-worker mode.

## Run contract

The CLI registers each run from its preprocessing manifest and an explicit
corpus path. Optional candidate-evaluation artifacts are registered as one
manifest-backed bundle. Registration validates required files and schemas before
the server starts.

A run descriptor contains:

- a user-facing label and stable run ID;
- preprocessing manifest path and fingerprint;
- corpus path and fingerprint;
- preprocessing definition identity;
- paths to `results.parquet`, `candidates.parquet`, `step_facts.parquet`, and
  `rejections.parquet`;
- optional candidate-evaluation manifest, `candidate_membership.parquet`, and
  `candidate_results.parquet`;
- semantic coordinates needed to reject invalid comparisons.

The first implementation accepts registered paths only at process startup. This
keeps filesystem authority at the CLI boundary and avoids turning API parameters
into a filesystem browser. A naked analysis summary or candidate-results file is
not a run: it cannot establish the joins or denominators needed by the waterfall.

## DuckDB model

The database schema is created by a small, explicit migration runner:

- `schema_migrations(version, applied_at)`
- `runs(run_id, label, descriptor_json, manifest_sha256, corpus_sha256,
  definition_id, registered_at)`
- `annotations(corpus_sha256, sample_id, decoder_output_sha256, verdict, note,
  created_at, updated_at)`
- `tags(tag_id, name, created_at)` with a unique normalized name
- `annotation_tags(corpus_sha256, sample_id, decoder_output_sha256, tag_id)`

The annotation primary key is based on corpus, sample, and exact decoder output,
not run ID. A conclusion therefore follows an unchanged example across baseline
and candidate runs but cannot silently attach to changed text. Annotation writes
use short explicit transactions and upserts. The DuckDB file is ignored by Git;
deterministic exports are the portable and reviewable representation.

Parquet paths remain data values at the registration boundary. Query construction
must not interpolate user filters, identifiers, or search text into SQL. The
backend exposes named analytical operations rather than general query execution.

## Query contracts

The query layer returns JSON-ready, typed read models for:

- available runs and their provenance;
- single-run waterfall stages with counts, denominators, rates, and units;
- terminal failure groups;
- paginated examples for a stage, failure group, or transition;
- one complete example with raw output, candidates, facts, rejections, and
  current annotation;
- compatible-run comparison stages and outcome transitions;
- the tag vocabulary and annotation mutations.

Waterfall stage definitions live in one backend module and have stable IDs,
labels, units, ordering, denominator rules, and drill-down predicates. The same
definitions drive both aggregates and example selection so a displayed count and
its list cannot drift apart.

The first version computes aggregates directly. It should add cached or
materialized summaries only after measurements show repeated scans are a
problem; any future cache key must include all relevant manifest fingerprints
and query parameters.

## HTTP and CLI boundary

The API is intentionally small:

- `GET /api/runs`
- `GET /api/runs/{run_id}/waterfall`
- `GET /api/runs/{run_id}/failures`
- `GET /api/runs/{run_id}/examples`
- `GET /api/runs/{run_id}/examples/{sample_id}`
- `GET /api/compare?baseline=...&candidate=...`
- `GET /api/tags`
- `POST /api/tags`
- `PUT /api/annotations/{corpus_sha256}/{sample_id}/{decoder_output_sha256}`
- `DELETE /api/annotations/{corpus_sha256}/{sample_id}/{decoder_output_sha256}`
- `GET /api/annotations/export`

The Typer CLI accepts one or more named run descriptors, a DuckDB path, host, and
port. The host defaults to `127.0.0.1`; non-loopback hosts and worker counts other
than one are outside this tool's contract. A representative invocation is:

```bash
uv run dr-code viewer \
  --run baseline=/path/to/baseline/run.json \
  --run candidate=/path/to/candidate/run.json \
  --database .runs/dr-code-viewer.duckdb
```

The concrete descriptor format may use a small JSON file to keep the CLI usable
when evaluation artifacts are present. It must remain explicit and validated.

## Frontend migration

The preprocessing-analysis package remains the React/Vite frontend, but its
runtime data source becomes the FastAPI client. Replace the long static report
with app-level navigation for Waterfall and Review; Compare appears whenever two
compatible runs are selected.

Remove the checked `viewer-data.json`, generated failure shards, snapshot sync
scripts, and byte-for-byte snapshot checks after dynamic feature parity is in
place. Keep reusable presentation components such as code blocks, diffs, and
status badges. Loading, empty, validation-error, and backend-error states are part
of the UI contract.

Review interactions save immediately and display pending, saved, and failed
states. Creating a tag adds it to the shared vocabulary and selects it for the
active example. Keyboard and form semantics must support rapid sequential
annotation without sacrificing accessible labels and focus behavior.

## Annotation export and regression handoff

The export endpoint returns stable ordering and excludes machine-local paths and
timestamps that would create noisy diffs. It includes the example identity,
verdict, note, and sorted tags. Promotion from reviewed annotations into the
golden hard-example suite is a separate explicit operation; viewing or editing an
annotation never mutates test fixtures automatically.

This PR establishes the annotation workflow and portable export. The parsing
refactor PR can consume selected `should_be_parseable` examples as named golden
cases and list them in its implementation plan.

## Implementation sequence

1. Add DuckDB, FastAPI, and Uvicorn dependencies; define and test run descriptors,
   artifact validation, schema migrations, annotation storage, and stable exports.
2. Implement and test DuckDB queries for a single-run waterfall, grouped
   failures, drill-down examples, and compatible before/after transitions using
   small realistic Parquet fixtures.
3. Add the Typer entry point and FastAPI application with typed request/response
   models, loopback defaults, static frontend serving, and API tests.
4. Replace frontend snapshot imports with a typed API client and implement
   Waterfall, Compare, and Review, including verdicts, notes, existing-tag
   selection, and tag creation.
5. Remove obsolete generated viewer data and sync machinery, update usage
   documentation, and exercise the app against the full adjacent generation
   corpus artifacts.

## Verification and acceptance criteria

The change is complete when:

- a single command starts one local process and serves the application;
- the real generation-corpus run produces waterfall and failure totals consistent
  with its authoritative artifacts;
- selecting every aggregate opens examples governed by the same predicate;
- compatible runs yield stable deltas and transitions, while incompatible runs
  are rejected with a useful explanation;
- annotation verdicts, notes, and tags survive restart in DuckDB;
- identical samples reuse annotations across runs and changed decoder output does
  not;
- annotation export is deterministic;
- no static corpus snapshot or failure-detail shard remains in the frontend;
- Python formatting, linting, typing, and tests pass;
- frontend typechecking, tests, and production build pass.

The architecture intentionally leaves out authentication, remote access,
multi-process writes, arbitrary SQL, generic dashboard configuration, automatic
golden-fixture mutation, and speculative aggregate caching.
