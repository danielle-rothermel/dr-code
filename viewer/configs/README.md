# Viewer run configs

These descriptors register the complete local artifact bundles currently
available in the adjacent `gen-viewer` checkout. Paths are relative to each
descriptor, so they work when `dr-code` and `gen-viewer` are sibling
directories.

## Available runs

- `generation-corpus-functions-v1-20260719.json` registers the generation
  corpus, its complete functions-v1 preprocessing artifacts, and the completed
  candidate evaluation.
- `generation-corpus-functions-v1-20260719-subprocess-v3-20260722.json`
  registers the same baseline preprocessing artifacts with the append-only
  subprocess-v3 candidate evaluation. It is retained as diagnostic evidence.
- `generation-corpus-functions-v1-extraction-redesign-v2-subprocess-v3-20260722.json`
  registers the append-only extraction-redesign-v2 preprocessing artifacts and
  their subprocess-v3 candidate evaluation. It is superseded diagnostic
  evidence from before the additive-salvage and runner-protocol corrections.
- `generation-corpus-functions-v1-20260719-subprocess-v4-20260722.json`
  registers the definitive host-subprocess baseline evaluation.
- `generation-corpus-functions-v1-extraction-redesign-v4-subprocess-v4-20260722.json`
  registers the definitive extraction redesign, exact salvage-boundary
  provenance, and host-subprocess evaluation. Matching results are
  deterministically reused from the definitive baseline.

Start the viewer from the repository root:

```bash
uv run dr-code viewer \
  --run generation-corpus=viewer/configs/generation-corpus-functions-v1-20260719.json \
  --database .runs/dr-code-viewer.duckdb
```

Compare the subprocess-v3 baseline and extraction redesign runs with:

```bash
uv run dr-code viewer \
  --run baseline=viewer/configs/generation-corpus-functions-v1-20260719-subprocess-v3-20260722.json \
  --run extraction-redesign-v2=viewer/configs/generation-corpus-functions-v1-extraction-redesign-v2-subprocess-v3-20260722.json \
  --database .runs/dr-code-viewer.duckdb
```

Compare the definitive baseline and extraction redesign runs with:

```bash
uv run dr-code viewer \
  --run before=viewer/configs/generation-corpus-functions-v1-20260719-subprocess-v4-20260722.json \
  --run after=viewer/configs/generation-corpus-functions-v1-extraction-redesign-v4-subprocess-v4-20260722.json \
  --database .runs/dr-code-viewer.duckdb
```

The directories suffixed `failed-cleanup-race` and
`failed-pre-cid-contract` are intentionally not registered. They contain
interrupted evaluation database state but no completed manifest, membership
Parquet, or results Parquet, so they do not satisfy the viewer's run contract.
