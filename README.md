# dr-code

Research harness for the compression–correctness evaluation pipeline: given a natural-language description of a HumanEval function, can a decoder reconstruct working Python, and how compressible is the description?

Design docs: [docs/plans/README.md](docs/plans/README.md)

**Status:** Stage 1 (generation & dataset) is complete. Stage 2 (parsing) is next — see [Stage 2 handoff](docs/plans/stage-02-handoff.md).

## Stage 1 demo

Full verification: pool import, live fresh generation (or offline stub), export, stats, and side-by-side spot check.

```bash
# Offline smoke (no API key)
uv run scripts/demo_stage1.py --skip-live

# Live generation (requires OPENROUTER_API_KEY)
uv run scripts/demo_stage1.py
```

Exports land in `exports/demo/pool.parquet` and `exports/demo/fresh.parquet`.

## Stage 1 CLIs

Import pool artifacts into unified `AttemptRecord` exports:

```bash
uv run scripts/import_pool_attempts.py --help
```

Generate fresh decoder attempts via dr-providers:

```bash
uv run scripts/generate_decoder_attempts.py --list-profiles
uv run scripts/generate_decoder_attempts.py \
  --profile openrouter/google/gemini-3.1-flash-lite/off/v1 \
  --task-ids HumanEval/0 --stats
```

Rebuild the offline HumanEval+ snapshot (requires network):

```bash
uv run scripts/build_humaneval_snapshot.py
```

## Tests

```bash
uv run pytest tests/unit -q
```
