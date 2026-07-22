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

This plan addresses both problems as one pipeline redesign. It also incorporates
the completed annotation audit described below; those dispositions are inputs
to the design, not claims about the current implementation.

## Annotation audit and approved dispositions

The authoritative review input is the 2026-07-22 annotation checkpoint
`.runs/checkpoints/annotations-pr58-input-20260722.json`, whose source content
has SHA-256
`c9ebe01e398bfe589fe67b69260553dc37647a8d9a662463cda5c169eb75a441`.
It contains 101 sample records covering 91 distinct decoder-output hashes.
Slice 1 must promote this ignored working checkpoint into a checked-in,
content-addressed immutable fixture plus a manifest recording its source hash,
corpus hash, sample identities, decoder-output hashes, verdicts, notes, and
tags. Tests and reports must consume the immutable fixture, never mutable
viewer annotation state. The canonical annotation-record stream in that
fixture has SHA-256
`0048761890b9e20af9016d15f7b4eacaeb2171bfd895579b89293650063437d5`;
the source-file hash and canonical-record hash protect different boundaries and
both belong in the manifest.

The final review disposition is:

| Disposition | Records | Distinct outputs | Contract |
| --- | ---: | ---: | --- |
| Approved recovery | 34 | 32 | Produce and evaluate at least one candidate through an approved path. |
| Intrinsically invalid task answer | 1 | 1 | Preserve and inspect the complete source, but accept its compiler rejection and do not add a semantic repair. |
| Enforced negative | 62 | 54 | Produce no final function candidate. |
| Explicit annotation/definition contract conflict | 4 | 4 | Quarantine the review conflict and preserve candidates containing top-level helper functions. |

The 34 approved recovery records are partitioned by mechanism. Full sample IDs
are included so implementation, differential reports, and regenerated
artifacts use the same target set:

- **Additive last-complete-return salvage (17):**
  `0006045dfcf5c7342a8861c7e4df8bb156136955da6f6db95dd20771cc90a13c`,
  `000abb3b0cf414898dec45e01f24d77555f740dd6a6afc61bb717a161ab1e5ff`,
  `000b2d2a6d3a95b76c8f5d3856473ff4748266477944e3e4f209e8ace85e11c6`,
  `002586cfadacbd3c8c90ad1458cfeb1ddac3e98eb992d3c1c1cef6b2d6ecb97f`,
  `002e8ca69992e6f2476cd633449c7d6e1e698beac40754ea3bddeb33a3736479`,
  `0045d5846f3cdb60b986e48a7497d4e6cd8d1b596bf86787eec29ce418b82c9d`,
  `005e4dbb92546f66c00fe50a9b8c2a60b1e6236229cae977b7c470825f491220`,
  `007496b5ede4838498fbbbb013e329912f8c9c9975a5c9e858a32b5bdd9e245b`,
  `009229fa44589710da4167501b298904cb1ebeb540eb61c5ef31cf9cc92d7cf5`,
  `00a28b65a6d999a169dd63f8970fd0669059ab4fb198a4612624faf64819673c`,
  `00acbed430da279d3218815857e40d145306347c71f7b74710dafcbdce150057`,
  `03bf1a71c60c1ec965d0811ac003d2eda037d2f41ffec5d00164a14d322b1d40`,
  `043ecf8df114fbe041a8cfe5fcc861ac9bade4fda687e1e51363bf0f8b936834`,
  `044257059b7cdd429c32ef93a1a82bff756d5970f1df09cf3aba1e5285c03a99`,
  `05c5730f1a28dc826a25049271651244bb67b7c3e2870e7f248c0aa15f452455`,
  `944eceb0403a8aaaa59952d2104c4b394c27a200ddea2f116872ded1a2946323`,
  and
  `94cf8b1702329d7039300d8d95f4b719598f20df7e678dd58e1ac924793b1a11`.
- **Fenced JSON (10 records, 9 outputs):**
  `002653e5bedcc96bf64400ee829fcf6cfa11390067d32b54d6bcaa2852fce845`,
  `005112a4d5ba95009ad10b1f902fd68873d0b8b0b3e473f7f5800fd5aebd3d9f`,
  `00b784107b21f23d31bc1bcfb612ee5ae68bc07bdc35b1b72a10af90492b2667`,
  `013acbb7523823ed997a6d8f9202ac4c5454bca044e2f876f8e4c78b18ba2caf`,
  `02984bfb239c28d3fab1f937e72f27bb6b0607fee54e716c21e8353c06565117`,
  `050b6c02fc87937b14262daeef4bc4afb4e5580c59d9a40a9decb58622ec9279`,
  `064546520976db7310f6603564f891b8a0ad3fa0fe5376f3d4151295e29b43e8`,
  `06c75934f6496e36be5f9f54882a9f8f7c4db4c3c2bb4cf11b16aa265f283ce8`,
  `0950f6d06aa8f5a91932eae1fce3c146f39a4ac9730b95a784d972ffa342d02a`,
  and
  `0a11ee04cc5d6730eaf5b31bf47369794ee08c0217628e7e039eeff09bd3359d`.
  The `005112...` and `0950...` records intentionally share one decoder
  output and must receive the same result.
- **Singleton string sequence (1):**
  `06d4696832183551eb47dbdc3a85b700b4e29b89b05356b8e12f292940c2fe0e`.
- **EOF-only JSON-envelope closure (2):**
  `0969d4ae3f03ead63ddf5bb6d0e92987f2b5ae8de6c85084c26703a992cc4dc0`
  and
  `1d1c04d7ba17329208238b7644e73b5184b0d10f1fc7fbdda5d7e55b89d06a04`.
- **Top-level lambda to function (3 records, 2 outputs):**
  `0462724e63bb6624b916e014e73961d0ac9c1349b56552d91171a44a7e8dd55a`,
  `215466665c0e83972dbcab694790a0472507102240ecbb95c8d3e6a73fc11f01`,
  and
  `9418eb2c35631d4facec78c23fe5706c57c218ae7997fe72768c6d30ed528d6e`.
  The first two records intentionally share one decoder output.
- **Additive eligible unfenced segment (1):**
  `e3f0dc1201762d620d34eeae26af23a469ce08b345764cbc52b5d8d1782bef9c`.

Sample
`007a142c27d875e3697b4cbf449d02febd47a71bd539e46aa6234e3f4bd4f011`
is the single intrinsically invalid answer. Removing destructive physical-line
return truncation preserves its complete source for exact inspection, but
Python compilation rejects its assignment expression because it attempts to
rebind the comprehension iteration variable `total`. It therefore remains an
ordinary `no_compilable_candidate` outcome. That compiler result is not
permission to invent a semantic rewrite or to validate only a source prefix.

The enforced-negative set is exactly the 62 immutable snapshot records whose
verdict is `expected_no_code`, excluding the following four class-tagged
contract-conflict sample IDs:

- `92ff75341c6dd4dac580b09c886ebd75fc45eecd3ade800b684831e2ba94f34c`;
- `9476f3c1b91432ab782dc33aa0e818ae427148050717573e1d64c513c1454a7a`;
- `960e6e2c777a759e3464e764e951956b0f7be92dc935b0863aa45e693c67be46`;
- `967b8fe0ca2001e94110e5e4f8faafcebf87a9fb439992bf34bf447b56d6ff01`.

Those four are explicit annotation/definition contract conflicts rather than
ordinary extraction negatives. Each response contains class-oriented code but
also at least one top-level helper such as `main` or a demonstration function.
The human annotations requested no code, while the named definition
deliberately accepts any top-level function without imposing the benchmark's
expected function name. The definition contract wins: these cases must remain
candidate-producing but quarantined and reported as conflicts. No class method
is promoted, no class is converted, and "top level" is not weakened.

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
8. Make every approved recovery additive, retain the raw path, and preserve an
   auditable distinction between representation recovery, source salvage, and
   semantic evaluation.
9. Turn the annotation dispositions into an identity-aware hard contract and
   validate the frozen rules against a separate holdout before authoritative
   regeneration.

## Non-goals

- Do not impose a benchmark-specific function-name requirement during
  preprocessing.
- Do not recursively decode arbitrary nested representations to a fixed point.
- Do not introduce a generic plugin registry before multiple independent
  extension points require one.
- Do not add a process-global parse or source cache across samples.
- Do not perform arbitrary syntax completion, recursively balance malformed
  delimiters, or repair task logic.
- Do not add operator normalization. The two records initially tagged
  `character_replacement` already contain ASCII `<=` and `==`; their corruption
  is caused by physical-line return truncation. In particular, do not translate
  `<=`, `==`, Unicode ellipsis, or Unicode mathematical operators on the basis
  of rendered appearance.
- Do not expand or reinterpret the existing general Unicode-normalization
  contract in this slice; any future lexical Unicode-safety change requires its
  own corpus audit and versioned migration.
- Do not promote class methods, nested functions, or arbitrary callable
  expressions into top-level function candidates.
- Do not combine other unrelated candidate-cleaning changes with this redesign.
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
       additive candidate salvage
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

Other audited paths include:

```text
normalized_raw_response
  -> close_json_envelope_at_eof(appended='"}')
  -> top_level_json_code
  -> anchored_python_block

normalized_raw_response
  -> singleton_string_sequence(kind=list, index=0)
  -> anchored_python_block

normalized_raw_response
  -> top_level_bare_lambda(name=candidate)
  -> lambda_to_function

normalized_raw_response
  -> anchored_python_block
  -> last_complete_return_prefix(end_line=..., end_column=...)
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
4. field-marker code, when applicable;
5. the string in a whole-response singleton string sequence, when applicable;
   and
6. a string `code` value recovered by the narrowly defined EOF-only JSON
   envelope closure, when applicable.

A representation decoder adds a possible interpretation; it does not replace
the original text. Strict decoders always run before a recovery decoder.

The singleton sequence interpretation accepts only a complete top-level Python
literal shaped as a `list` or `tuple` with exactly one string element. It does
not accept multiple elements, nested containers, mappings, sets, bytes,
f-strings, calls, assignments, or an extracted string without an established
top-level Python anchor. Treat parser syntax and recursion failures as ordinary
non-matches and bound the input size before parsing. This rule recovers the
audited `06d469...` representation without turning arbitrary literal data into
a recursive representation language.

The EOF-only JSON recovery runs only when strict whole-response JSON decoding
has failed. A bounded JSON lexical scanner must prove all of the following:

- the trimmed response begins one top-level object;
- the active value is that object's string member named `code`;
- EOF occurs inside that string;
- no JSON escape is dangling or partially formed; and
- closing the code quote and the one object delimiter is sufficient for
  `json.loads` to accept the entire repaired response.

The recovery may append only the missing quote and `}` at EOF. It must not
strip trailing text, close Python strings or delimiters, repair partial Unicode
escapes, search nested objects or arrays, or remove a literal truncation marker.
After closure, the ordinary top-level-mapping and string-`code` validation still
applies. The repaired envelope and decoded code retain a distinct ordered
provenance operation. In particular, `1d1c04...` remains incomplete Python;
ordinary additive salvage and validation decide whether any prefix is useful.

#### Text recovery

For each response representation, consider the original form followed by a
structurally recovered escaped-Python form when recovery is applicable. This
replaces `escaped_python` and `escaped_markdown_wrapper` as precomposed
discovery strategies.

#### Structural block extraction

Apply the rich fence extractor to each recovered text fragment:

- visit every fenced block and every eligible unfenced segment in document
  order; and
- retain the current first-eligible-unfenced behavior only within each
  contiguous unfenced segment.

The presence of any fence must not globally suppress an eligible unfenced
segment. This is required for `e3f0dc...`, whose valid function precedes a
closed Python fence containing only an example name guard. Raw fenced and
unfenced content remains eligible for ordinary candidate discovery. A tag is
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

Interpret a top-level lambda only after a response or block has yielded one
complete Python-shaped fragment. Accept exactly either:

- a module consisting of one bare lambda expression; or
- a module consisting of one simple name assigned one lambda.

Emit an additive function-form draft whose body returns the lambda expression's
body and whose arguments preserve the lambda's positional-only, positional,
keyword-only, variadic, and default arguments. Reuse the assignment target for
the named form. Use the stable synthetic name `candidate` for the bare
form and record that choice in provenance. Reject attribute or subscript
targets, multiple targets, annotated assignments, nested lambdas, lambdas mixed
with other statements, and transformations that cannot be represented without
changing closure semantics. Preserve the original lambda fragment as a raw
candidate; conversion does not replace it.

#### Python candidate discovery

For each interpreted block:

1. consider its original body;
2. consider Markdown-wrapper removal when it changes or reveals eligible text;
3. discover anchored Python regions using the established code-like rules; and
4. emit every nonblank candidate draft in deterministic order.

Keep `ExtractCandidates` as a thin adapter that invokes the engine, converts
drafts into the trace artifact, and reports structured counts and paths.

#### Additive last-complete-return salvage

Replace destructive `drop_after_last_return` cleaning with an additive salvage
operation. Always emit the original discovered candidate first. Then, when a
lexical scan proves that a real Python `return` token has a complete logical
statement boundary, optionally emit the prefix ending at the last such
boundary. A multiline return continues through its matching parentheses,
brackets, or braces and ends at its logical `NEWLINE`, not at the physical line
containing `return`.

The scanner must distinguish code from comments, ordinary strings, triple-
quoted strings and docstrings, honor escapes, and never react to identifiers
such as `return_value`. If EOF or malformed tokenization prevents proof of a
complete return statement, emit no salvaged draft. Do not append delimiters or
otherwise make the prefix compile inside this operation. The later inspection
stage validates the exact complete salvaged source. Record the retained end
line and column and the salvage operation in its path.

This additive rule is the approved explanation for the 17 listed recoveries,
including the two records originally mis-tagged as character replacement. It
also preserves the complete `007a142...` source for exact inspection instead
of turning its multiline `return any(...)` into a different invalid prefix;
the original source's compiler rejection remains authoritative.

### 4. Candidate cleaning boundary

The extraction engine should remove structural outer fences before emitting a
candidate. After parity tests prove that emitted candidates cannot retain those
fences, remove the downstream `strip_fences` step as obsolete.

Leave the following cleaning operations unchanged during this stage:

- dedenting;
- smart-quote normalization;
- name-guard splitting;
- import-line repair;
- missing-import inference; and
- import deduplication.

Remove destructive trimming after the last physical `return` line once the
additive salvage operation and exact differential tests are in place. No
operator-normalization step replaces it.

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

The parse-once guarantee is exact and full-source: after cleaning and
deduplication, each distinct candidate source is passed once, in its entirety,
to the shared parse-and-compile helper. Inspection must not accept a parseable
prefix, ignore trailing text, or recompile a rewritten copy. Instrument the
helper in tests and assert exactly one call for every unique candidate ID and
zero calls from all downstream policy filters. Bounded parsing used earlier to
recognize an external response representation or a top-level lambda shape is a
separate extraction-boundary operation; it must not call the candidate
validation helper or substitute for full-source inspection.

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

Writers emit only the new versioned schema. Analysis and viewer readers must
dispatch explicitly by schema version and support both the existing flat
`variant`/`strategy` artifacts and the new ordered-path artifacts for the
defined migration window. Keep that compatibility at the persistence-reader
boundary: do not add dual fields to new artifacts and do not restore flat
origins to the domain model. Pin one old viewer-data fixture and one new fixture
and prove that both render equivalent legacy facts, while new path operations
render only for the new schema. Unknown schema versions fail with an actionable
error rather than being guessed.

### Analysis

Update analysis to support:

- initial contribution by extraction operation and complete path;
- final contribution by operation and complete path;
- convergence across distinct paths;
- `fenced_json_code` recovery and test-success rates; and
- readable path rendering in deterministic spot checks.

Add explicit before/after comparison tables keyed by sample ID, candidate ID,
and decoder-output hash. They must distinguish candidate membership changes,
source changes, outcome changes, new or converged paths, evaluation changes,
and unchanged rows. Every delta must be attributable to an approved operation
or be reviewed as a regression. Old/new reader compatibility is temporary and
versioned; new writers and the in-memory domain use only the new schema.

### Definition versioning

Because this stack has not merged, keep the public
`humaneval-function-candidates@v1` coordinate while bumping every changed step
version and accepting a new definition hash. If the definition is treated as
released before implementation begins, introduce `v2` instead.

## Implementation sequence

Each slice should leave the repository green and reviewable.

### Slice 1: Characterization

- Promote the 101-record annotation checkpoint into the immutable,
  content-addressed fixture and manifest described above; assert 91 distinct
  output hashes and the complete disposition reconciliation.
- Add the fence and representation fixture matrix.
- Record existing candidate-source and ordering behavior.
- Record existing failure codes, failed steps, facts, and output kinds.
- Add all exact approved-recovery inputs, the intrinsic-invalid input, and the
  four annotation/definition contract conflicts as named regression fixtures.
- Combine the 91 annotation-output cases with the 21 synthetic contract cases
  into one 112-case hard fixture. Partition by decoder-output hash, never by
  sample ID, using the recorded `sha256-prefix-mod-5-v1` rule: 90 development
  cases and 22 holdout cases. Persist the deterministic assignment in the
  manifest so duplicate outputs cannot appear in both partitions. The 62
  enforced-negative records remain a required subset across the two
  partitions. Do not inspect or tune rules against the holdout until the
  implementation and hard-suite rules are frozen.
- Capture a deterministic representative corpus subset for differential tests.

### Slice 2: Rich fence extraction

- Add the structured fence document and block models.
- Implement structural extraction with parity tests.
- Visit eligible fenced and unfenced segments additively in document order,
  including the exact `e3f0dc...` regression.
- Migrate shared text analysis, text transforms, preprocessing, and legacy
  HumanEval cleaning.
- Remove `split_by_fences` after all callers migrate.
- Preserve behavior and metric identities.

### Slice 3: Modular extraction with parity

- Add text fragments, candidate drafts, and ordered extraction paths.
- Implement response representation, recovery, block extraction,
  interpretation, and discovery functions.
- Add the bounded singleton-string-sequence and EOF-only JSON-envelope
  interpretations behind exact positive and negative characterization tests.
- Add the conservative top-level bare/named lambda-to-function interpretation,
  including the stable `candidate` name and explicit rejected shapes.
- Make `ExtractCandidates` a thin adapter.
- Update trace and artifact provenance schemas.
- Demonstrate parity before enabling each new interpretation.

### Slice 4: Fenced-JSON recovery

- Enable `fenced_json_code` using tag-or-shape identification plus strict JSON
  and mapping validation.
- Cover tagged, untagged, malformed, misleading, multiple, and nested-fence
  cases.
- Remove downstream fence stripping only if its obsolescence is proven.
- Confirm the intended differential behavior on the representative corpus
  subset.

### Slice 5: Additive source salvage

- Add lexical last-complete-return discovery with logical-statement,
  string/comment/docstring, escape, and malformed-input coverage.
- Emit the raw candidate first and a provenance-distinct prefix only when its
  boundary is proven.
- Remove destructive physical-line `drop_after_last_return` only after exact
  old/new tests cover all 17 approved records and `007a142...` retains its full
  source for the expected compiler rejection.
- Confirm that no operator-normalization behavior was introduced.

### Slice 6: Parse-once inspection

- Add inspection models and the inspected candidate-set artifact.
- Add the inspection step after deduplication.
- Refactor the four policies to consume stored inspections.
- Add the final conversion back to `CodeCandidateSetArtifact`.
- Preserve rejection and terminal-failure semantics.
- Instrument and prove one exact full-source parse-and-compile call per unique
  cleaned candidate ID.

### Slice 7: Analysis and viewer migration

- Update Parquet schemas and mechanical projection.
- Update origin and convergence analysis.
- Bump writer schemas, add explicit old/new reader dispatch, and update path
  rendering.
- Add identity-keyed before/after candidate, outcome, provenance, and evaluation
  comparisons.
- Update report generation, documentation, and viewer tests.

### Slice 8: Frozen-rule holdout

- Freeze extraction and salvage rules after the development hard suite passes.
- Run the sealed holdout without tuning; its positive and synthetic contracts
  must pass, and all held-out annotated negatives must remain without a final
  function candidate.
- Reconcile all 101 records: all 34 approved recoveries reach evaluation, the
  intrinsic-invalid record remains an attributed compile rejection, all 62
  enforced negatives stay negative, and all four quarantined contract-conflict
  records remain candidate-producing under the definition's top-level-function
  policy.
- Treat any unexpected result as a rule-design failure requiring a documented
  change, a new frozen assignment, and a complete hard-suite and holdout rerun.

### Slice 9: Authoritative append-only regeneration and rescore

- Create a new append-only preprocessing run ID.
- Reprocess the complete corpus from the final source commit.
- Create a new append-only candidate-evaluation run and immutable manifest.
- Rescore every candidate membership, not only changed rows, under the pinned
  HumanEval+ snapshot and OCI image.
- Create a new append-only analysis run from those exact preprocessing and
  evaluation manifests; never overwrite the old preprocessing, evaluation,
  analysis, viewer, or annotation artifacts.
- Regenerate compact tables, summary, report, viewer data, and viewer snapshot.
- Produce the complete old/new comparison and require every outcome,
  membership, source, and test-result delta to be attributed or rejected.
- Re-run concurrency and infrastructure-failure checks.

## Verification gates

### Fence extraction

- backtick and tilde fences;
- tagged and untagged fences;
- multiple fenced and unfenced segments;
- matching and mismatched closers;
- unclosed fences;
- blank fenced bodies;
- eligible code before, between, and after fenced blocks;
- a fence containing only usage text without suppression of neighboring
  unfenced code;
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
- JSON code containing a Python fence;
- convergence between raw and derived paths;
- all 10 audited fenced-JSON records, including the repeated decoder output;
- a whole-response singleton `list` or `tuple` containing one code string;
- singleton-sequence negatives for multiple or nested elements, mappings,
  bytes, f-strings, calls, assignments, and strings without a code anchor;
- both exact EOF-truncated JSON records and a complete-escape representative;
- EOF-recovery negatives for dangling backslashes, partial `\u` escapes,
  nested or non-string `code`, missing-code objects, already closed strings,
  trailing text, and any repair requiring more than the quote and one `}`;
- exact bare and named lambda conversions, stable synthetic naming, preserved
  arguments and defaults, and rejection of every excluded lambda shape;
- exact additive raw-before-derived ordering for every interpretation; and
- exact `e3f0dc...` recovery from an unfenced segment adjacent to a fenced usage
  block.

### Additive salvage and lexical safety

- all 17 approved last-complete-return records produce the expected source and
  ordered salvage path;
- multiline returns retain their complete logical expression through matching
  parentheses, brackets, and braces;
- `return` text in comments, ordinary strings, triple-quoted strings,
  docstrings, and identifiers does not establish a boundary;
- incomplete tokenization or an unclosed return expression emits no salvaged
  draft and never fabricates a closer;
- raw candidates remain before salvaged candidates and convergence is stable;
- the full `007a142...` source reaches inspection and retains its exact
  `no_compilable_candidate` outcome without prefix acceptance; and
- no new transform changes `<=`, `==`, `≤`, or `…` in code, strings,
  docstrings, or comments; existing general Unicode normalization remains a
  separately versioned contract.

### Candidate identity and inspection

- cleaning preserves paths and clears stale IDs after source changes;
- deduplication assigns IDs from cleaned source and merges paths;
- inspection invokes the shared validator exactly once per unique cleaned
  candidate ID and parses and compiles the exact full source;
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
- origin paths project and reload without loss;
- old flat-origin artifacts load only through the old schema reader, new path
  artifacts load only through the new schema reader, and both render correctly;
- new writers never emit the old flat fields or a dual schema; and
- unknown reader schema versions fail explicitly.

### Annotation contract, hard suite, and holdout

- the immutable fixture hash, 101 record count, 91 output count, corpus hash,
  and every sample/output identity match its manifest;
- the fixture contains 112 cases total, the recorded
  `sha256-prefix-mod-5-v1` partition recomputes exactly 90 development and 22
  holdout cases, and no decoder-output hash crosses partitions;
- dispositions reconcile to 34 approved recoveries over 32 outputs, one
  intrinsic invalid, 62 enforced negatives over 54 outputs, and four
  annotation/definition contract conflicts;
- output-hash grouping gives duplicate records one disposition and one
  hard-suite/holdout assignment;
- every approved recovery has at least one provenance path naming only its
  approved operations and reaches candidate evaluation;
- all 62 enforced negatives have no final function candidate after the frozen
  implementation;
- all four annotation/definition conflicts remain quarantined,
  candidate-producing cases because their top-level helper functions satisfy
  the definition, with no class method promotion;
- the hard suite passes before rule freeze and the sealed holdout passes after
  it without tuning; and
- `007a142...` remains an attributed compiler rejection and is not inserted
  into candidate-evaluation membership.

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
- the new append-only run evaluates and reports every candidate membership,
  not only changed candidates;
- no final infrastructure failures remain; and
- all regenerated manifests and provenance coordinates match the final source
  and immutable execution inputs;
- old run directories and manifests remain byte-for-byte unchanged;
- the before/after comparison accounts for every sample, candidate, outcome,
  origin path, and evaluation row with no unexplained delta;
- generated Parquet, JSON, report, viewer-data, and snapshot artifacts match
  their manifest hashes, schema versions, row counts, and join invariants; and
- rerunning regeneration from the same manifests produces identical content
  hashes.

## Expected outcome

The completed redesign should make extraction behavior a visible composition
of small deterministic, additive operations rather than a matrix of
preassembled strategies. It should satisfy the immutable annotation contract,
recover the approved fenced JSON, singleton sequence, EOF envelope, lambda,
unfenced-segment, and last-return cases through named paths, and preserve the
enforced negatives while explicitly reporting the annotation/definition
conflicts. It should retain meaningful convergence and before/after analysis,
reduce exact full-source Python parsing and compilation from as many as four
validation calls per surviving candidate
to one, and regenerate the complete preprocessing/evaluation/analysis stack
append-only without collapsing the existing policy-stage failure contract.
