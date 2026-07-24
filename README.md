# dr-code

## Python execution

`dr_code.execution.run_python_subprocess` runs Python source in a fresh
`sys.executable -I` process with bounded text input, a shared stdout/stderr
limit, a wall-clock deadline, and process-group cleanup. HumanEval uses this
primitive through an injectable batch-runner interface.

This execution boundary provides no operating-system containment. Candidate
code has the worker's filesystem, credential, process, and network permissions.
Process-group cleanup cannot guarantee termination of descendants that detach
from the group. Run evaluations only on disposable workers whose permissions,
network access, resources, and lifetime are constrained externally.

## Behavioral mutants

`dr_code.mutants` creates deterministic, execution-validated behavioral
mutants from pinned HumanEval+ canonical programs and inputs. It publishes
stable JSONL plus an authenticating manifest as one immutable directory. The
authenticated snapshot ships in the wheel as the offline default; `--hf`
explicitly selects the independent pinned Hugging Face source.

```bash
uv run python -m dr_code.mutants generate \
  --dry-run \
  --tasks HumanEval/0
```

See [`docs/behavioral-mutants.md`](docs/behavioral-mutants.md) for the five
operator families, acceptance gates, artifact contract, and generation
command.

## Corpus analysis and viewer

Completed preprocessing runs are immutable manifest-backed bundles containing
`results.parquet`, `candidates.parquet`, `step_facts.parquet`, and
`rejections.parquet`. Candidate evaluations may add their own complete
manifest plus membership and result Parquets.

Generate compact summaries and schema-pinned comparison relations with:

```bash
uv run python scripts/analyze_preprocessing_corpus.py \
  --dataset-id evalplus/humanevalplus \
  --corpus /data/corpus.parquet \
  --run-dir /data/preprocessing/run-id \
  --candidate-evaluation /data/evaluations/run-id \
  --output-dir analysis/preprocessing/run-id

uv run python scripts/compare_preprocessing_runs.py \
  --dataset-id evalplus/humanevalplus \
  --corpus /data/corpus.parquet \
  --before-run /data/preprocessing/before \
  --after-run /data/preprocessing/after \
  --output-dir analysis/preprocessing-comparisons/before--after
```

The local viewer queries those external Parquets dynamically and stores only
run registrations, tags, and example annotations in DuckDB. See
[`viewer/README.md`](viewer/README.md) for the descriptor contract, build, and
loopback-only serving instructions. Published wheels include the complete
frontend, so an installed `dr-code viewer` does not depend on a repository
checkout or a separate static-assets directory.

The frontend workspace and its frozen pnpm lockfile are the source of truth for
those packaged assets. `python3 scripts/build_viewer_assets.py` updates the
checked archive, and `python3 scripts/build_viewer_assets.py --check` rebuilds
in a temporary workspace and fails if either checked artifact is stale.

Classify a descriptor's parse and measured candidate-test failures through an
explicit subscription provider/model pair with:

```bash
uv run dr-code classify-failures /data/run.json \
  --provider glm-coding \
  --model glm-4.6 \
  --repeats 5
```

The command checkpoints strict canonical JSONL details under a full
experiment SHA-256 covering the authenticated run, provider transport, model,
repeat and aggregation policy, taxonomy, prompt template, and selection
limits. An explicit `--details` path is permanently owned by that exact
experiment; `--force` recomputes it without allowing cross-experiment
replacement. Machine task rollups are published as one transaction after the
details artifact is complete. Human rows remain protected, and an all-failed
rerun removes only stale machine rows owned by the same experiment and selected
task scope.

See [`docs/failure-classification.md`](docs/failure-classification.md) for the
artifact, resume, and publication contracts plus the live validation checklist.
