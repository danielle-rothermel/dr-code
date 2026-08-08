# Legacy HumanEval generation corpus

`scripts/build_legacy_humaneval_corpus.py` reconstructs a provenance-complete
HumanEval generation corpus from a raw `dr-llm` pool-dump directory. It reads
the dump manifest and compressed pool rows directly; the lossy per-task
Parquet splits are not an input.

The adapter publishes three files only after validating the complete build:

- `legacy-humaneval-generation-corpus.parquet` uses the 25-column
  generation-corpus schema. One row represents one historical decoder
  attempt, including repeated attempts with identical output text.
- `legacy-humaneval-generation-requests.parquet` is a one-to-one sidecar keyed
  by `sample_id`. It preserves exact serialized requests, model and prompt
  configuration IDs, reasoning controls, sampling controls, encoder lineage,
  generation mode, and character-budget classification.
- `legacy-humaneval-generation-corpus.manifest.json` records source and output
  hashes, row and task counts, classification counts, prompt fidelity, and
  encoder-reference coverage.

Run it with an explicit empty destination:

```bash
uv run python scripts/build_legacy_humaneval_corpus.py \
  /path/to/raw-pool-dump \
  --output-dir /path/to/new-output-directory
```

The default HumanEval snapshot is
`tests/corpus/humanevalplus_snapshot.json`. It is used only to supply the
semantic task prompt for migrated rows whose persisted request explicitly
says it is unavailable. Those rows are labeled `semantic_only`; persisted
requests that can be projected without alteration are labeled
`exact_request`.

## Reconstruction rules

Decoder pools are identified from their pool schema and decoded with the
source dump's persisted HumanEval and output-path hints. A canonical sample ID
is derived from the source project, pool, and sample ID rather than output
text, so exact duplicate generations remain separate attempts.

Rows whose metadata identifies an encoder sample are joined to the matching
encoder pool row. The adapter verifies that the encoder response, model
configuration, and prompt template agree with the lineage embedded in the
decoder row. Encoder and decoder prompts come from the exact persisted request
message arrays; the request sidecar retains the complete JSON even when the
canonical system/user projection is unavailable.

Character budgets are parsed independently from the encoder template ID and
rendered prompt. If both are present they must agree. Direct rows are
`no_budget`. A row that claims encoder lineage but does not identify an
encoder source is retained as `unresolved_encoder` with an extraction warning
rather than being mislabeled as direct or encoder/decoder.

`is_retry` means the source pool needed more than one attempt. `is_partial`
means the persisted finish reason is `length`. The sidecar keeps the exact
source attempt count and finish reason.

## Validation boundary

Before publishing, the builder verifies:

- canonical and sidecar schemas;
- unique corpus and sidecar sample IDs with one-to-one coverage;
- complete encoder model/output fields for every `enc_dec` row;
- canonical content hashes for every row;
- source-manifest dump completeness;
- complete resolution of every explicit encoder reference; and
- agreement between prompt and template character budgets.

The output directory must be new or empty. Failed builds remain in a temporary
directory and do not replace or mix with existing artifacts.

This command reconstructs data but does not execute model-produced code.
Candidate execution remains restricted to a disposable worker under the
HumanEval execution contract.
