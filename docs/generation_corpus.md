# Generation corpus extraction

The generation corpus extracts archived model activity into validated,
analysis-ready Parquet tables. Extraction preserves source lifecycle records
separately from generation records and never executes generated code.

## Generation corpus bundle

Each build atomically publishes one **generation corpus bundle**: a directory
of validated Parquet tables plus a build manifest.

| Artifact | Term | Grain |
|---|---|---|
| `source_records.parquet` | source lifecycle record | Every source row selected for the dataset, including blank, failed, seeded, and pending rows. |
| `generations.parquet` | generation record | One row per persisted nonblank final generation output. Equal output text remains separate when it came from separate source lifecycle records. |
| `encoder_artifacts.parquet` | encoder artifact record | Nonblank encoder outputs that are not linked to a generation record. |
| `requests.parquet` | request provenance record | Exactly one request and config provenance row per generation record. |
| `tasks.parquet` | task record | Content-addressed task material resolved from the explicit task source. |
| `manifest.json` | build manifest | Corpus adapter identity, pool-dump manifest identity, row counts, schemas, and SHA-256 artifact hashes. |

The input **pool dump** also carries its own `manifest.json` (source manifest)
naming the archived gzip tables. Do not confuse that source manifest with the
build manifest published by the corpus build.

The writer validates unique identities, generation/request one-to-one joins,
source and task foreign keys, encoder lineage, record content hashes, schemas,
and row counts before publication. A failed build leaves no partial destination.

## Build command

```bash
uv run python scripts/build_generation_corpus.py DATASET \
  --dump-directory DUMP_DIRECTORY \
  --task-source TASK_SOURCE \
  --output-directory OUTPUT_DIRECTORY
```

`DATASET` is one of:

- `human_eval`
- `mbpp_pro`
- `humaneval_pro`
- `class_eval`
- `bigcodebench_lite_pro`
- `nl_latents`

The dump directory must contain the audited pool-dump `manifest.json` and its
named gzip tables. The output directory must be new or empty. Task sources are
explicit:

| Dataset | Task source |
|---|---|
| HumanEval | The pinned HumanEval+ snapshot JSON file. |
| MBPP Pro | The pinned `CodeEval-Pro__mbpp-pro/train/v3` cache directory. |
| HumanEval Pro | The pinned `CodeEval-Pro__humaneval-pro/train/v3` cache directory. |
| ClassEval | The pinned `FudanSELab__ClassEval/test/v1` cache directory. |
| BigCodeBench Lite Pro | The pinned `CodeEval-Pro__bigcodebench-lite-pro/train/v3` cache directory. |
| NL Latents | The archived NL Latents root containing the primary, seed-41, and workshop task trees. |

For example:

```bash
DUMP_DIRECTORY=/path/to/20260621_manual
OUTPUT_DIRECTORY=/path/to/corpora/human_eval
TASK_SOURCE=/path/to/humanevalplus_snapshot.json

uv run python scripts/build_generation_corpus.py human_eval \
  --dump-directory "${DUMP_DIRECTORY:?}" \
  --task-source "${TASK_SOURCE:?}" \
  --output-directory "${OUTPUT_DIRECTORY:?}"
```

## Audited populations

The production build validates the source manifest, pinned task input, and
dataset-specific invariants. The complete 2026-06-21 dump produces:

| Dataset | Source lifecycle records | Generation records | Standalone encoders | Request provenance records | Task records |
|---|---:|---:|---:|---:|---:|
| HumanEval | 630,089 | 203,407 | 221,084 | 203,407 | 164 |
| MBPP Pro | 143,655 | 22,639 | 111,631 | 22,639 | 375 |
| HumanEval Pro | 62,543 | 9,848 | 48,607 | 9,848 | 163 |
| ClassEval | 37,564 | 5,934 | 29,196 | 5,934 | 196 |
| BigCodeBench Lite Pro | 20,712 | 3,262 | 16,082 | 3,262 | 108 |
| NL Latents | 192,333 | 191,462 | 526 | 191,462 | 294 |

ClassEval and BigCodeBench contain two content-distinct source variants per
logical task, so their task-record counts are twice their logical task counts.

## Evidence and re-evaluation boundary

Prompt and config fidelity is explicit: exact persisted requests,
task prompts recovered from pinned task material, and unavailable request
evidence remain distinguishable. Raw `attempt_count` is preserved, but the
extractor does not invent per-attempt indexes or retry identities.

The corpus preserves the task material and generation output needed to analyze
every dataset. Re-evaluation additionally requires a compatible evaluator and
an explicit immutable runtime. NL Latents task records deliberately set
`execution_ready` to false: the archive contains Python, Java, and Rust
generation output but does not pin the required language toolchains, evaluator
adapters, resource policy, or containment boundary. Code candidates for
evaluation are extracted separately from generation output during
preprocessing.
