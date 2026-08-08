# Previous HumanEval data

The previous-generation HumanEval data is a local, read-only snapshot. It is
not committed to this repository. The authoritative source dump is:

```text
/Users/daniellerothermel/drotherm/data/code-comp/
  dr-llm-humaneval-pool-dumps/20260621_manual/
```

The recommended cleaned representation for current `dr-code` analysis is:

```text
/Users/daniellerothermel/drotherm/data/.codex/dr-code/2026-08-08/
  legacy-humaneval-corpus-reviewed/
```

Treat the raw pool dumps as the provenance authority. Every other artifact in
this document is derived from them for a narrower analysis purpose.

## Cleaned canonical corpus

`legacy-humaneval-corpus-reviewed/` was built from the raw pool dumps by
`scripts/build_legacy_humaneval_corpus.py`. It contains:

- `legacy-humaneval-generation-corpus.parquet`: 203,407 historical decoder
  attempts in the canonical 25-column generation-corpus schema;
- `legacy-humaneval-generation-requests.parquet`: a one-to-one sidecar with
  the exact persisted requests, prompt and model configuration IDs, reasoning
  and sampling controls, budget classification, and encoder lineage; and
- `legacy-humaneval-generation-corpus.manifest.json`: source and output
  hashes, schemas, counts, prompt fidelity, and encoder-reference coverage.

The builder selects exact HumanEval code outputs from the decoder-like pools,
joins each explicit encoder reference back to its source pool row, verifies
the embedded lineage and rendered character budget, and projects the result
into the current corpus schema. It uses the checked-in HumanEval snapshot only
for rows whose persisted request explicitly says the original prompt is
unavailable.

The verified result contains 181,328 encoder/decoder rows, 22,078 direct rows,
and one row labeled `unresolved_encoder`. All 180,924 distinct explicit encoder
references resolve. The source contains 163 HumanEval tasks; `HumanEval/32` is
absent. See [Legacy HumanEval generation corpus](legacy_humaneval_corpus.md)
for the schema and reconstruction contract.

## Source snapshot and earlier derivatives

The `20260621_manual/` directory contains the source snapshot and several
older cleaned views. Their derivation and intended use differ:

| Artifact | Contents | Derived from | Transformation and limitations |
| --- | --- | --- | --- |
| `manifest.json` and 26 `*.jsonl.gz` files | 1,086,896 persisted pool rows with requests, responses, metadata, keys, and extraction hints | The historical `code_comp_t1`, `code_comp_v0`, and `nl_latents` `dr-llm` pool projects | `dump_humaneval_candidate_pool_rows.py` streamed each selected pool without normalizing it and recorded pool schemas, row counts, and original running state. This is the highest-fidelity source. |
| `policy_summary.json` | A 26-pool inventory of schema and HumanEval-policy matches | The same live historical pool projects | `explore_humaneval_pool_policy.py` inspected the pools before extraction. It is an audit summary, not a row corpus. |
| `humaneval_code_attempts.parquet` | 203,407 nonblank HumanEval decoder/direct outputs across 163 tasks, with model and source metadata plus the decoder input description | The 26 raw pool dumps | `extract_humaneval_code_attempts.py` applied the exact `human_eval/HumanEval/<n>` identity policy, excluded HumanEvalPro and other datasets, and extracted decoder code text. For 9,856 migrated rows it backfilled the official task prompt from the old `nl-code` cache. Encoder rows and exact serialized requests are not retained as first-class rows. |
| `humaneval_code_attempts_preview.csv` | A small human-readable preview | `humaneval_code_attempts.parquet` | Convenience export only; never use it as a corpus source. |
| `per_elem/` | 163 task Parquets, 163 exact-output-count JSONL files, and a manifest; 203,407 total rows and 172,454 task-local unique raw outputs | `humaneval_code_attempts.parquet` | `split_humaneval_attempts_by_task.py` partitioned the unified table by task. Each `*-dedup.jsonl` collapses identical raw output strings to `{out, count}` and therefore loses attempt provenance. The task Parquets preserve the unified table but do not recover encoder requests. |
| `rich_trace_split/clean/full_encoder_chain.jsonl.gz` | 181,328 decoder rows paired with resolved encoder rows | The raw pool dumps | `split_rich_trace_candidate_rows.py` required rendered encoder and decoder prompts and outputs plus valid encoder lineage. This is a high-fidelity encoder/decoder subset, not the full decoder corpus. |
| `rich_trace_split/messy/messy_decoder_candidates.jsonl.gz` | 274,376 decoder-like rows with exclusion reasons | The same rich-trace classification pass | Retains rows outside the clean rule for audit. The largest reasons are missing decoder output, missing decoder prompt, and missing encoder source kind. |
| `rich_traces/rich_trace_attempts.parquet` | 181,328 flattened full encoder-chain rows with prompts, outputs, configuration IDs, provider/model data, usage, cost, and latency | `rich_trace_split/clean/full_encoder_chain.jsonl.gz` | `extract_rich_trace_rows.py` normalized the clean pairs. It excludes direct, docstring-only, migrated official-decoder, response-only, and unresolved rows by design. |
| `rich_traces/by_dataset/` | `human_eval/all.parquet`, 163 per-task Parquets, and a manifest | `rich_traces/rich_trace_attempts.parquet` | `split_rich_traces_by_dataset_task.py` produced analysis partitions without changing row content. |

The simple extraction scripts and their original reproduction commands are
documented in the adjacent `dr-llm` checkout at
`docs/humaneval-pool-extraction.md`. The rich-trace scripts and documentation
are preserved in historical `dr-llm` commit `07ca4b3`; they are not part of
the current `dr-code` implementation.

## Which artifact to use

- Use the canonical corpus plus request sidecar for new preprocessing,
  sampling, or evaluation work in `dr-code`.
- Use the raw gzip dumps when auditing lineage or adding a field the canonical
  projection does not expose.
- Use `humaneval_code_attempts.parquet` only to reproduce the earlier broad
  output/description analyses.
- Use `per_elem/` only to reproduce task-local raw-output inspection or the
  old exact-dedup sampling workflow.
- Use `rich_traces/` only to reproduce analyses explicitly restricted to
  complete encoder/decoder chains.

None of these data transformations executes generated code. Historical
candidate evaluation must still run on a disposable worker.
