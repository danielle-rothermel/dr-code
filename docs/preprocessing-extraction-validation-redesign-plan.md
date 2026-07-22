# Candidate Extraction and Parse-Once Validation Redesign Plan

## Status

Proposed implementation plan. This document defines the intended architecture,
migration sequence, and verification gates; it does not describe an implemented
state.

## Context

The `humaneval-function-candidates@v1` flow currently combines response
representation recovery, Markdown fence handling, candidate discovery, and
provenance construction inside `ExtractCandidates`. Its four named extraction
strategies are precomposed paths:

- fenced blocks;
- Markdown-wrapper removal;
- escaped-Python recovery; and
- escaped-Python recovery followed by Markdown-wrapper removal.

That arrangement makes transformation order implicit and requires each new
recovery mechanism to be manually combined with existing mechanisms. The flat
`CandidateOrigin(variant, strategy)` contract also cannot describe a complete
path through several transformations.

Corpus spot check
`160d0e739ebc363ca524d674279de890596ff7b69e408baa710f10ec0933ab2d`
demonstrates the limitation. Its decoder response is a JSON object inside a
`json`-tagged Markdown fence. The current flow extracts the JSON object as a
candidate but does not apply top-level JSON-code decoding after fence
extraction. Python accepts the object as a dictionary literal, so the
plain-literal filter rejects it even though the object's `code` field contains
a valid function. A corpus diagnostic found 194 fenced JSON objects with string
`code` fields among the 198 `plain_literal_only` samples; 176 of those code
fields produce valid function candidates when reprocessed.

Candidate validation has a related composition problem. The plain-literal,
code-representation, compilation, and top-level-function filters each call the
same parse-and-compile helper. A candidate that reaches the final filter may be
parsed and compiled four times. Consolidating the filters into one opaque step
would remove that duplication but would also discard useful filter-stage
failure semantics.

This plan addresses both problems as one pipeline redesign.

## Goals

1. Separate response recovery, structural block extraction, block
   interpretation, and Python candidate discovery.
2. Preserve fence metadata and allow additional interpretations of fenced
   content without suppressing the raw-content path.
3. Record complete, ordered extraction provenance that survives cleaning and
   deduplication.
4. Parse and compile each unique cleaned candidate once per preprocessing
   invocation.
5. Preserve the current ordered validation policies, terminal failure codes,
   failed-step identities, rejection details, final candidate ordering, and
   public `CodeCandidateSetArtifact` output.
6. Keep extraction deterministic, total over arbitrary text, and bounded.
7. Regenerate all authoritative corpus, evaluation, analysis, and viewer
   artifacts from explicit immutable inputs after the semantic change.

## Non-goals

- Do not impose a benchmark-specific function-name requirement during
  preprocessing.
- Do not recursively decode arbitrary nested representations to a fixed point.
- Do not introduce a generic plugin registry before multiple independent
  extension points require one.
- Do not add a process-global parse or source cache across samples.
- Do not combine unrelated candidate-cleaning changes with this redesign.
- Do not infer preprocessing outcomes in the corpus projector or analyzer.

## Target pipeline

```text
TextArtifact
  -> text normalization
  -> modular candidate extraction
       response representations
       text recovery
       structural block extraction
       block interpretation
       Python candidate discovery
  -> candidate cleaning
  -> remove blank candidates
  -> deduplicate cleaned sources and assign candidate IDs
  -> inspect each unique candidate once
  -> ordered policy filters using stored inspection
  -> CodeCandidateSetArtifact
```

The boundary between the two redesign stages is deliberate:

- extraction owns where candidate text came from;
- cleaning establishes the candidate's canonical source;
- deduplication establishes content-derived identity; and
- inspection owns what that canonical Python source structurally contains.

## Stage 1: Modular candidate extraction

### 1. Rich fence extraction

Replace `split_by_fences` with a total structural extractor that returns an
ordered document representation. A fenced segment should retain at least:

```python
FenceBlock(
    index=0,
    marker="```",
    tag="json",
    content='{"code": "..."}',
    closed=True,
)
```

The structural extractor must:

- preserve segment and fenced-block order;
- preserve the normalized fence tag;
- distinguish backtick and tilde markers;
- record whether the opening fence was closed;
- preserve existing matching-closer behavior;
- retain unfenced segments;
- accept arbitrary text without raising; and
- make no claim about whether block content is Python, JSON, or prose.

Keep `FENCE_LINE_RE` unchanged so metric operators that include the regex in
their recorded identity do not silently change. Migrate all callers in
preprocessing, shared text analysis, text transforms, and the legacy HumanEval
cleaning path before removing `split_by_fences`.

### 2. Extraction-domain values

Introduce small internal values that keep source and provenance aligned:

- `TextFragment`: text plus its ordered extraction path;
- `FenceBlock`: the structural Markdown representation;
- `CandidateDraft`: candidate source plus its complete origin path; and
- `ExtractionOperation`: one typed operation in that path.

An origin must be able to express paths such as:

```text
normalized_raw_response
  -> fenced_block(tag=json, index=0)
  -> fenced_json_code
  -> anchored_python_block
```

Candidate source and provenance should travel in one value internally rather
than in parallel arrays. The step adapter may convert drafts into the public
candidate-set artifact at its boundary.

### 3. Explicit extraction layers

Move domain logic out of the preprocessing step adapter into focused pure
functions with an explicit order.

#### Response representations

Produce the following additive representations in stable order:

1. normalized raw response;
2. decoded whole-response JSON string, when applicable;
3. string `code` value from a whole-response top-level JSON object, when
   applicable; and
4. field-marker code, when applicable.

A representation decoder adds a possible interpretation; it does not replace
the original text.

#### Text recovery

For each response representation, consider the original form followed by a
structurally recovered escaped-Python form when recovery is applicable. This
replaces `escaped_python` and `escaped_markdown_wrapper` as precomposed
discovery strategies.

#### Structural block extraction

Apply the rich fence extractor to each recovered text fragment:

- when fences are present, visit every fenced block in source order;
- otherwise, preserve the current first-eligible-unfenced-block behavior.

Raw fenced content remains eligible for ordinary candidate discovery. A tag is
evidence for an additional interpretation, not an exclusive content type.

#### Block interpretation

For each block, emit the raw body first. A fenced block is also eligible for a
`fenced_json_code` interpretation when either:

- its tag is `json`, case-insensitively; or
- its trimmed content begins with `{` and ends with `}`.

The tag and brace checks are only candidate-identification signals. Acceptance
requires `json.loads` to succeed, the decoded value to be a top-level mapping,
and its `code` member to be a string. Invalid or incompatible JSON produces no
derived interpretation and is not an operational error.

Do not search nested mappings or arrays for a `code` field. Process every
qualifying fenced block in order.

The decoded code string may re-enter structural candidate discovery once so a
JSON `code` value containing a Python fence remains supported. Do not permit
unbounded recursive representation decoding.

#### Python candidate discovery

For each interpreted block:

1. consider its original body;
2. consider Markdown-wrapper removal when it changes or reveals eligible text;
3. discover anchored Python regions using the established code-like rules; and
4. emit every nonblank candidate draft in deterministic order.

Keep `ExtractCandidates` as a thin adapter that invokes the engine, converts
drafts into the trace artifact, and reports structured counts and paths.

### 4. Candidate cleaning boundary

The extraction engine should remove structural outer fences before emitting a
candidate. After parity tests prove that emitted candidates cannot retain those
fences, remove the downstream `strip_fences` step as obsolete.

Leave the following cleaning operations unchanged during this stage:

- dedenting;
- smart-quote normalization;
- name-guard splitting;
- trimming after the last return;
- import-line repair;
- missing-import inference; and
- import deduplication.

Cleaning operations preserve origin paths. A transform that splits one
candidate copies its origin path to every result and clears any pre-cleaning
candidate ID.

### 5. Candidate identity and convergence

Continue assigning candidate IDs only after cleaning and nonblank filtering.
Deduplication must:

- keep the first exact cleaned source;
- assign the current content-derived candidate ID;
- merge every distinct extraction path in deterministic order; and
- report duplicate groups using complete paths.

A converged candidate remains one cleaned candidate reached through more than
one distinct origin path.

## Stage 2: Parse-once candidate inspection

### 1. Inspection boundary

Insert candidate inspection after nonblank filtering and deduplication. This
placement avoids parsing blank candidates, candidates that cleaning will still
change, and duplicate cleaned sources.

`inspect_candidate(source)` calls `validate_python_source_with_ast` exactly
once and derives all policy inputs from the returned validation and AST:

```python
CandidateInspection(
    validation=PythonSourceValidation(...),
    is_plain_literal_module=False,
    is_code_repr_assignment=False,
    top_level_function_names=("add",),
    top_level_async_function_names=(),
)
```

The AST is an ephemeral implementation detail of inspection. Store only typed,
serializable derived facts.

### 2. Inspected candidate artifact

Add an explicit inspected-candidate artifact containing aligned, typed values:

- cleaned source;
- candidate ID;
- complete origin paths; and
- `CandidateInspection`.

The inspection step accepts `CodeCandidateSetArtifact` and emits
`InspectedCandidateSetArtifact`. It records one inspection fact per candidate
ID so corpus projection remains mechanical.

An explicit artifact is preferred over ambient caching because it preserves
determinism, serialization, and step contracts. It is preferred over one
combined validation step because the existing policy stages remain useful
public trace semantics.

### 3. Ordered validation policies

Refactor the existing filters to consume and return inspected candidate sets
without parsing. Preserve their order:

1. plain-literal module;
2. `code = "..."` representation assignment;
3. parseable and compilable Python; and
4. at least one top-level synchronous or asynchronous function.

Each filter must preserve:

- candidate source, order, identity, and origin paths;
- its existing candidate rejection reason;
- its existing terminal failure code;
- its existing failed-step identity;
- parse and compile diagnostics in rejection facts; and
- function facts for survivors and no-function rejections.

The filter implementations consult inspection fields only. They must not call
`ast.parse`, `compile`, or the shared validation helper.

### 4. Public output conversion

After all policies pass, a final return step converts surviving inspected
candidates back to `CodeCandidateSetArtifact`, preserving source, order,
candidate ID, and complete provenance. The public preprocessing output contract
therefore remains unchanged.

## Persistence and analysis migration

### Provenance schema

Replace the candidate Parquet origin structure's flat `variant` and `strategy`
fields with an ordered extraction-path representation. Bump the projected
artifact schema and analysis/viewer payload schema explicitly.

The corpus projector must continue to copy trace-owned facts mechanically. It
must not parse source, identify extraction paths, or infer outcomes.

### Analysis

Update analysis to support:

- initial contribution by extraction operation and complete path;
- final contribution by operation and complete path;
- convergence across distinct paths;
- `fenced_json_code` recovery and test-success rates; and
- readable path rendering in deterministic spot checks.

Do not preserve `variant/strategy` as a compatibility layer unless an external
consumer is identified. Prefer one explicit schema transition.

### Definition versioning

Because this stack has not merged, keep the public
`humaneval-function-candidates@v1` coordinate while bumping every changed step
version and accepting a new definition hash. If the definition is treated as
released before implementation begins, introduce `v2` instead.

## Implementation sequence

Each slice should leave the repository green and reviewable.

### Slice 1: Characterization

- Add the fence and representation fixture matrix.
- Record existing candidate-source and ordering behavior.
- Record existing failure codes, failed steps, facts, and output kinds.
- Add the exact fenced-JSON spot check as a regression fixture.
- Capture a deterministic representative corpus subset for differential tests.

### Slice 2: Rich fence extraction

- Add the structured fence document and block models.
- Implement structural extraction with parity tests.
- Migrate shared text analysis, text transforms, preprocessing, and legacy
  HumanEval cleaning.
- Remove `split_by_fences` after all callers migrate.
- Preserve behavior and metric identities.

### Slice 3: Modular extraction with parity

- Add text fragments, candidate drafts, and ordered extraction paths.
- Implement response representation, recovery, block extraction,
  interpretation, and discovery functions.
- Make `ExtractCandidates` a thin adapter.
- Update trace and artifact provenance schemas.
- Demonstrate parity before enabling the new fenced-JSON interpretation.

### Slice 4: Fenced-JSON recovery

- Enable `fenced_json_code` using tag-or-shape identification plus strict JSON
  and mapping validation.
- Cover tagged, untagged, malformed, misleading, multiple, and nested-fence
  cases.
- Remove downstream fence stripping only if its obsolescence is proven.
- Confirm the intended differential behavior on the representative corpus
  subset.

### Slice 5: Parse-once inspection

- Add inspection models and the inspected candidate-set artifact.
- Add the inspection step after deduplication.
- Refactor the four policies to consume stored inspections.
- Add the final conversion back to `CodeCandidateSetArtifact`.
- Preserve rejection and terminal-failure semantics.

### Slice 6: Analysis and viewer migration

- Update Parquet schemas and mechanical projection.
- Update origin and convergence analysis.
- Bump viewer-data schema and update path rendering.
- Update report generation, documentation, and viewer tests.

### Slice 7: Authoritative regeneration

- Create a new append-only preprocessing run ID.
- Reprocess the complete corpus from the final source commit.
- Create a new candidate-evaluation run and immutable manifest.
- Evaluate every changed membership under the pinned HumanEval+ snapshot and
  OCI image.
- Regenerate compact tables, summary, report, viewer data, and viewer snapshot.
- Re-run concurrency and infrastructure-failure checks.

## Verification gates

### Fence extraction

- backtick and tilde fences;
- tagged and untagged fences;
- multiple fenced and unfenced segments;
- matching and mismatched closers;
- unclosed fences;
- blank fenced bodies;
- tag and order preservation; and
- arbitrary malformed text without exceptions.

### Extraction parity and intended changes

- exact candidate source and order for all established fixtures;
- whole-response JSON strings and objects;
- field-marker payloads;
- escaped Python and escaped fences;
- Markdown wrappers and anchored code blocks;
- exact fenced-JSON regression;
- tagged and brace-identified fenced JSON;
- malformed JSON and valid JSON without a string `code` field;
- multiple JSON fences;
- JSON code containing a Python fence; and
- convergence between raw and derived paths.

### Candidate identity and inspection

- cleaning preserves paths and clears stale IDs after source changes;
- deduplication assigns IDs from cleaned source and merges paths;
- inspection runs once per unique cleaned candidate;
- all policy filters perform zero parsing and compilation;
- parser stack overflow and recursion overflow remain data rejections;
- unrelated `MemoryError` still propagates;
- syntax warnings remain stable;
- async and nested-only functions retain current classifications; and
- final output remains `CodeCandidateSetArtifact`.

### Trace and persistence

- all new artifacts round-trip through trace serialization;
- filter failure codes and failed-step identities remain stable;
- rejection rows preserve policy-stage names and diagnostics;
- candidate rows receive inspection facts by candidate ID;
- the corpus projector never parses or infers source facts; and
- origin paths project and reload without loss.

### Repository checks

- focused extraction, preprocessing, trace, and corpus tests;
- full Python test suite;
- Ruff;
- ty;
- viewer typecheck, build, and tests; and
- deterministic regeneration checks.

### Full-corpus reconciliation

- exactly 365,216 source rows remain represented;
- present, missing, and blank denominators reconcile;
- every successful candidate is nonblank, compilable, and function-bearing;
- every final candidate has an ID and at least one complete origin path;
- every old-to-new outcome change is attributable to an intended extraction
  rule;
- the 198 prior `plain_literal_only` samples are specifically reconciled;
- candidate memberships fully join evaluation results;
- no final infrastructure failures remain; and
- all regenerated manifests and provenance coordinates match the final source
  and immutable execution inputs.

## Expected outcome

The completed redesign should make extraction behavior a visible composition
of small deterministic operations rather than a matrix of preassembled
strategies. It should recover fenced JSON code through a named provenance path,
retain meaningful convergence analysis, and reduce Python parsing and
compilation from as many as four calls per surviving candidate to one without
collapsing the existing policy-stage failure contract.
