# Preprocessing viewer frontend

React/Vite frontend for the local preprocessing viewer service. Runtime data is
loaded from the service's typed `/api` endpoints; this package does not contain
or synchronize corpus snapshots.

The application provides three views:

- **Waterfall** traces sample counts through preprocessing and opens the exact
  examples behind a stage.
- **Compare** shows compatible run deltas and inspectable terminal transitions.
- **Review** pages through complete terminal-failure examples as stacked cards
  and immediately saves verdicts, comments, and tags to the local annotation
  database.

## Run the complete local application

From the repository root, register one or more explicit run descriptors with the
viewer command:

```bash
uv run dr-code viewer \
  --run baseline=/path/to/baseline/run.json \
  --run candidate=/path/to/candidate/run.json \
  --database .runs/dr-code-viewer.duckdb
```

The command binds to loopback and serves both the API and built frontend. Run
paths are registered at startup; they are never supplied by the browser.

## Frontend development

Run Vite separately when iterating on the UI. The development server proxies
`/api` to the local service (default `http://127.0.0.1:8000`):

```bash
cd viewer
pnpm install --frozen-lockfile
DR_CODE_VIEWER_API_URL=http://127.0.0.1:8000 \
  pnpm --filter @dr-code/preprocessing-analysis dev
```

Verify the package with:

```bash
pnpm --filter @dr-code/preprocessing-analysis typecheck
pnpm --filter @dr-code/preprocessing-analysis test
pnpm --filter @dr-code/preprocessing-analysis build
```
