# Candidate Extraction and Parse-Once Validation Redesign Plan

## Status

The modular extraction, additive salvage, parse-once validation, versioned
provenance, host-subprocess evaluation, stdout-protocol hardening, and
deterministic evaluation-reuse work is implemented. The detailed architecture,
sample dispositions, and verification gates below remain the approved contract
and are preserved as the implementation record.

The append-only preprocessing-v3 run remains useful diagnostic evidence but is
superseded because its salvage provenance omitted the required `end_line` and
`end_column` boundary. The corrected preprocessing-v4, baseline and redesigned
candidate-evaluation-v4, analysis, and comparison artifacts are complete and
reconciled below.

## Pre-implementation context

Before this redesign, the `humaneval-function-candidates@v1` flow combined response
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

Candidate validation had a related composition problem. The plain-literal,
code-representation, compilation, and top-level-function filters each call the
same parse-and-compile helper. A candidate that reaches the final filter may be
parsed and compiled four times. Consolidating the filters into one opaque step
would remove that duplication but would also discard useful filter-stage
failure semantics.

The implementation addressed both problems as one pipeline redesign. It also
incorporates the completed annotation audit described below as an immutable
behavior contract.

## Annotation audit and approved dispositions

The authoritative review input is the 2026-07-22 annotation checkpoint
`.runs/checkpoints/annotations-pr58-input-20260722.json`, whose source content
has SHA-256
`c9ebe01e398bfe589fe67b69260553dc37647a8d9a662463cda5c169eb75a441`.
It contains 101 sample records covering 91 distinct decoder-output hashes.
Slice 1 promoted this ignored working checkpoint into a checked-in,
content-addressed immutable fixture whose contract records its source hash,
corpus hash, sample identities, decoder-output hashes, verdicts, notes, and
tags. Tests consume the immutable fixture, never mutable viewer annotation
state. The canonical annotation-record stream in that fixture has SHA-256
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

## Implemented pipeline

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

## Implemented evaluation and reuse boundary

### Host subprocess

HumanEval execution now uses a fresh host subprocess for each request:

```text
[sys.executable, "-I", "-c", runner_source]
```

`run_python_subprocess` provides bounded JSON input, a combined stdout/stderr
output bound, a finite positive deadline, a fresh process session, and
process-group termination. Candidate evaluation records
`subprocess:python-isolated@v1` and fingerprints the Python executable, host
platform, installed distributions, and trusted sources as execution
coordinates.

Production evaluation performs no per-candidate OCI invocation, image pull, or
image preflight. This is intentionally a host-process boundary, not an
operating-system sandbox: candidate code retains the worker's filesystem and
network permissions. Evaluation therefore belongs on a disposable,
constrained worker without valuable credentials or data.

### Stdout protocol

The dependency-free batch runner keeps its original stdout handle only for the
final JSON result. Before support or candidate code executes, Python-level
`sys.stdout` and `sys.__stdout__` are redirected to bounded stderr. Tests cover
top-level and function prints, support-code prints, `sys.__stdout__`, output
floods, malformed JSON, invalid result shapes, unknown case IDs, and duplicate
case IDs.

Direct writes to file descriptor 1 remain an explicit limitation. They can
corrupt the protocol, and valid forged JSON is not authenticated. A separate
result channel or protocol authentication would be required to close that path.

### Deterministic `--reuse-results-from`

The candidate-evaluation CLI accepts repeatable completed evaluation
directories through `--reuse-results-from`. Before importing results, it
validates:

- complete source manifests and required final artifacts;
- exact snapshot, metric, operator, runner, execution, and host-runtime
  coordinates;
- candidate-results schema, row count, and SHA-256;
- candidate source SHA-256, task fingerprint, and recomputed evaluation key;
- profile, operator, record status, outcome, and result-value invariants; and
- agreement between duplicate reused keys.

Only keys in the target work set are imported; new keys execute normally.
Partial, incompatible, self-referential, duplicate, hash-mismatched, and
conflicting sources fail closed. The target manifest records ordered reuse
source hashes and per-source reuse counts. Those descriptors are immutable
resume coordinates, so changing their values or order rejects a partial resume
rather than silently changing its result set.

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

## Implementation sequence and status

All nine slices are implemented. The 135-test hard suite is the stable contract.
Slice 9 first produced preprocessing-v3 diagnostic evidence, then regenerated
preprocessing-v4 and every downstream artifact after review required exact
salvage-boundary details in provenance.

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
  into one sealed 112-case hard-suite cohort. Partition by decoder-output hash,
  never by sample ID, using the recorded `sha256-prefix-mod-5-v1` rule: 90
  development
  cases and 22 holdout cases. Persist the deterministic assignment in the
  manifest so duplicate outputs cannot appear in both partitions. The 62
  enforced-negative records remain a required subset across the two
  partitions. Do not inspect or tune rules against the holdout until the
  implementation and hard-suite rules are frozen.
- Keep later full-corpus discoveries outside that sealed assignment. The
  before/after transition artifact with SHA-256
  `9a6e4e88f3b1f672616b14cf9490604beeff7413a3508984e2bd10b2ada3b7b6`
  identifies 18 genuine extracted-to-no-compilable regressions. Promote them
  into an explicit post-holdout, full-corpus development cohort using exact
  candidates from the baseline candidate artifact with SHA-256
  `64d3effc33089e1fa36aa1db9ce0377e55cf3b324e1e8ab41105c0d99106e560`;
  do not relabel the three hash-assigned holdout members as unseen holdouts.
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
- After the sealed reconciliation is complete, keep post-holdout corpus
  regressions in their separately labeled development cohort. They may drive a
  repair, but cannot retroactively count as holdout evidence.

### Slice 9: Authoritative append-only regeneration and rescore

- Create a new append-only preprocessing-v4 run ID after the salvage
  `end_line`/`end_column` provenance contract and its hard fixtures are final.
- Reprocess the complete corpus from that final source commit.
- Create a new append-only candidate-evaluation run and immutable manifest.
- Rescore every candidate membership, not only changed rows, under the pinned
  HumanEval+ snapshot and the `subprocess:python-isolated@v1` host runner.
- Run the NumPy and canonical-solution preflight on the disposable evaluation
  worker before starting the score run. Do not restore the superseded OCI
  backend when regenerating these artifacts.
- Create a new append-only analysis run from those exact preprocessing and
  evaluation manifests; never overwrite the old preprocessing, evaluation,
  analysis, viewer, or annotation artifacts.
- Regenerate compact tables, summary, report, viewer data, and viewer snapshot.
- Produce the complete old/new comparison and require every outcome,
  membership, source, and test-result delta to be attributed or rejected.
- Re-run concurrency and infrastructure-failure checks.

The complete preprocessing-v3 directory must remain unchanged as superseded
diagnostic evidence. It is not an acceptable reuse source for final v4 scoring
because its preprocessing provenance contract is incomplete.

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
- the centralized fixture contains 130 cases: the original sealed 112-case
  cohort still recomputes exactly 90 development and 22 holdout cases under
  `sha256-prefix-mod-5-v1`, while all 18 later full-corpus regressions are
  explicitly post-holdout development cases; no decoder-output hash is reused;
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
- all 18 post-holdout regressions reproduce their baseline exact candidate
  sources and names through provenance paths containing the repaired
  `drop_after_last_return_salvage` operation; and
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

## Implementation evidence appendix

### Stable hard contract

The checked fixture
`tests/preprocessing/fixtures/hard_examples.json` has file SHA-256
`c31734a1003fed8fdfeafe30027f68ecc3446bce5668354b81b0bc42cdef5063`.
It contains 130 unique decoder-output contracts:

- 112 sealed cases: 90 development and 22 holdout under
  `sha256-prefix-mod-5-v1`;
- 18 separately labeled post-holdout full-corpus regressions;
- 101 annotation records covering 91 distinct outputs;
- 34 approved recoveries over 32 outputs;
- one intrinsic invalid;
- 62 enforced negatives over 54 outputs; and
- four quarantined annotation/definition conflicts.

`tests/preprocessing/test_hard_examples.py` collects 135 tests: 130
per-output pipeline cases plus five fixture-integrity, adjudication, and
provenance-oracle tests. All 135 pass with the exact v4 boundary-detail
provenance expectations.

### Superseded preprocessing-v3 diagnostic

The complete diagnostic run is
`generation-corpus-functions-v1-extraction-redesign-v3-20260722`. Its manifest
records:

- source commit
  `025a3f042507159b2b72e2eab03ffbaf4f292b43`;
- source diff SHA-256
  `905096d3b2768f8aec0a04c05de4bdd0732f924cebf0315cd5200db0c7632e88`;
- corpus SHA-256
  `a58acf1b1ed0ad54dc91d12bcca80398f3f3850b559f8051f52af2e4d4f1c4f5`;
- definition hash
  `b2da7cbd62c7702069afd750e92265255b9b7451fa59cc858f709aafba36848a3de17065307ec57594777b043f881d77111f8faeb013746bcf7aa2eb2575f436`;
- artifact schema version 2; and
- 365,216 input rows in 179 row groups.

Its outcomes reconcile exactly:

| Outcome | Rows |
| --- | ---: |
| `decoder_output_blank` | 109 |
| `decoder_output_missing` | 57,346 |
| `function_candidates_extracted` | 305,048 |
| `no_code_candidates` | 313 |
| `no_compilable_candidate` | 749 |
| `no_top_level_function_candidate` | 1,635 |
| `plain_literal_only` | 16 |
| **Total** | **365,216** |

Its projected relation totals are:

| Relation | Rows |
| --- | ---: |
| `results.parquet` | 365,216 |
| `candidates.parquet` | 433,412 |
| `rejections.parquet` | 104,133 |
| `step_facts.parquet` | 5,221,754 |

A read-only fixture-to-artifact audit found zero mismatches under the
then-current provenance contract:

- 110 corpus-backed fixture cases mapped to 275 exact result rows;
- all decoder-output hashes, outcomes, stable failure codes, and failed steps
  matched;
- successful source rows produced 154 candidates in exact source order;
- candidate function names and required or forbidden origin paths matched; and
- the remaining 20 cases were synthetic and had no authoritative corpus row.

These facts remain useful diagnostics, but v3 is not authoritative for final
comparison or scoring. Its salvage origin records name the operation without
the newly required `end_line` and `end_column`, so preprocessing-v4 must be
regenerated rather than relabeling or patching v3.

### Final preprocessing-v4 evidence

The authoritative run is
`generation-corpus-functions-v1-extraction-redesign-v4-20260722`. Its complete
179-row-group manifest has SHA-256
`0cbe708f469722f05bd714abe3978702a1f066b7e0be11399ce74c240db5c965`
and records source commit `b3cee2c7e0796809e9c8ba43d657ce876fc711ce`,
corpus SHA-256
`a58acf1b1ed0ad54dc91d12bcca80398f3f3850b559f8051f52af2e4d4f1c4f5`,
definition hash
`b2da7cbd62c7702069afd750e92265255b9b7451fa59cc858f709aafba36848a3de17065307ec57594777b043f881d77111f8faeb013746bcf7aa2eb2575f436`,
and `expand_last_return_salvage@4`.

Its counts equal v3 exactly: 365,216 results, 433,412 candidates, 104,133
rejections, 5,221,754 step facts, and 305,048 samples with final function
candidates. A row-level audit found zero fixture or annotation mismatches. All
189,564 salvage operations across 125,557 candidate rows carry exact boundary
coordinates; after removing only those new details, every v4 candidate row is
identical to v3 in source, order, names, inspection, and remaining lineage.

### Final candidate-evaluation evidence

Both evaluations use `subprocess:python-isolated@v1`, a null legacy
`sandbox_image`, the pinned HumanEval+ snapshot, and execution fingerprint
`1ccd695ab2c431f4d798979db14cf0b5a58df356ed3016cfa0ffda3093d7b6e5`.
Every result is measured and neither run contains an infrastructure failure.

| Coordinate | Baseline | Redesign |
| --- | ---: | ---: |
| Membership rows | 325,769 | 433,412 |
| Unique result rows | 216,527 | 316,618 |
| Reused baseline results | 0 | 216,527 |
| Newly executed results | 216,527 | 100,091 |
| Passing candidate rows | 245,839 | 305,622 |
| Samples with a passing candidate | 227,444 | 230,814 |
| Evaluation manifest SHA-256 | `18944b5479b851e8feeb02da361a1489f5ab9ce589ae1fd6e8e77bf8723f0177` | `a5cc78b2c66e3b17a1c59123d4bfe22d9dd4da35cbbee788eff07a68c33a9592` |

The redesigned manifest authenticates the reused source manifest and result
hash and imports all 216,527 matching keys. This makes shared evaluation
results deterministic: all 325,769 baseline result-bearing memberships retain
the same task, source, evaluation key, and result; 101,553 of those memberships
move to a different candidate index under the new additive ordering, 224,216
retain the same index, 107,643 are added, and none are removed.

### Final comparison and viewer evidence

The comparison summary has SHA-256
`17de77f53927ded217548d884d17dfa4c6513c9b7bce5e5af3c58dd5d320c1cd`;
its complete manifest has SHA-256
`9b134618c7a0ddcb6d41cebbced9047ff374495b862446398fbe76faaa3ecad8`.
It reconciles all 365,216 samples and records 4,885 preprocessing-outcome
changes with no decoder-output identity change.

The final hash-validated baseline and redesigned analysis summaries have
SHA-256
`2e3058308c0733e16897939b1b7724a9c9f4689b8b7fc8b223f602c6c26de35d`
and
`e8d3de1e33559c8cad06de63faa31a5c6a992b87f6cf199e0410ada5153879da`,
respectively. Both validate the manifest-published membership and result hashes
against the exact Parquet bytes and report no evaluation-linkage limitation.

The live viewer comparison reports these waterfall deltas:

| Stage | Baseline | Redesign | Delta |
| --- | ---: | ---: | ---: |
| Extracted response candidate | 307,398 | 307,448 | +50 |
| Compilable candidate | 302,149 | 306,683 | +4,534 |
| Top-level function candidate | 300,184 | 305,048 | +4,864 |
| Tested candidate | 300,184 | 305,048 | +4,864 |
| Passing candidate | 227,444 | 230,814 | +3,370 |

There are exactly 3,370 passing-sample gains and zero passing-sample losses.
Both committed viewer descriptors load, pass relational validation, and produce
this compatible comparison from the external append-only artifacts.

## Expected outcome

The implemented redesign makes extraction behavior a visible composition of
small deterministic, additive operations rather than a matrix of preassembled
strategies. It satisfies the immutable annotation contract, recovers the
approved fenced JSON, singleton sequence, EOF envelope, lambda,
unfenced-segment, and last-return cases through named paths, and preserves the
enforced negatives while explicitly reporting the annotation/definition
conflicts. Exact full-source Python parsing and compilation is reduced from as
many as four validation calls per surviving candidate to one without collapsing
the existing policy-stage failure contract.

The boundary-detail repair and the complete v4 preprocessing, evaluation,
analysis, comparison, and viewer stack are regenerated and reconciled
append-only.
