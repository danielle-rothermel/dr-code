# Preprocessing analysis viewer

Static Vite/React viewer for the checked preprocessing artifact. The app imports
its summary snapshot at build time and never reads the repository filesystem at
runtime. The all-failures explorer lazily fetches only packaged local static
indexes and detail shards under `public/data/failure-examples`; it does not call
an analysis service or fetch repository files.

Schema-v2 artifacts may include a joined `candidate_evaluation` block. When it
is present, the app also shows candidate execution outcomes, sample best-of
outcomes, preprocessing/evaluation comparisons, test-result examples, and the
manifest-backed execution profile and limitations. Older preprocessing-only
snapshots and early evaluation payloads without provenance remain supported.

## Refresh the snapshot

The source artifacts are
`analysis/preprocessing/generation-corpus-functions-v1-20260719/viewer-data.json`
and its sibling `failure-examples/` directory. After regenerating them, copy the
summary and a clean replacement of the packaged failure shards into this package
with:

```bash
cd viewer
pnpm --filter @dr-code/preprocessing-analysis data:sync
```

## Develop and verify

```bash
cd viewer
pnpm install --frozen-lockfile
pnpm --filter @dr-code/preprocessing-analysis dev
pnpm --filter @dr-code/preprocessing-analysis data:check
pnpm --filter @dr-code/preprocessing-analysis build
pnpm --filter @dr-code/preprocessing-analysis test
```

`data:check` and the package build compare the canonical and packaged summary
plus the complete failure-shard file tree byte-for-byte, catching missing, stale,
or changed snapshot files. The workspace-wide checks remain `pnpm typecheck`,
`pnpm build`, and `pnpm test`.
