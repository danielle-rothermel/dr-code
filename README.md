# dr-code

Research harness for the compression–correctness evaluation pipeline: given a natural-language description of a HumanEval function, can a decoder reconstruct working Python, and how compressible is the description?

Design docs: [docs/plans/README.md](docs/plans/README.md)

## Stage 1 demo

Run the stage 1 smoke demo (HumanEval+ loader, pool import, export round-trip, stats):

```bash
uv run scripts/demo_stage1.py
```

Import pool artifacts into unified `AttemptRecord` exports:

```bash
uv run scripts/import_pool_attempts.py --help
```

Rebuild the offline HumanEval+ snapshot (requires network):

```bash
uv run scripts/build_humaneval_snapshot.py
```

## Tests

```bash
uv run pytest tests/unit -q
```
