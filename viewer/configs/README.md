# Viewer run configs

These descriptors register the complete local artifact bundles currently
available in the adjacent `gen-viewer` checkout. Paths are relative to each
descriptor, so they work when `dr-code` and `gen-viewer` are sibling
directories.

## Available runs

- `generation-corpus-functions-v1-20260719.json` registers the generation
  corpus, its complete functions-v1 preprocessing artifacts, and the completed
  candidate evaluation.

Start the viewer from the repository root:

```bash
uv run dr-code viewer \
  --run generation-corpus=viewer/configs/generation-corpus-functions-v1-20260719.json \
  --database .runs/dr-code-viewer.duckdb
```

The directories suffixed `failed-cleanup-race` and
`failed-pre-cid-contract` are intentionally not registered. They contain
interrupted evaluation database state but no completed manifest, membership
Parquet, or results Parquet, so they do not satisfy the viewer's run contract.
