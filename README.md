# dr-code

## Python execution

Untrusted Python runs through the **dr-exec** package: a hermetic
`interpreter -I -c <source>` child with caller-declared budgets (wall clock,
output, input), an explicit environment grant, a per-run scratch working
directory, and race-safe process-group teardown. Outcomes are data — every
spawned run returns a `RunResult` carrying a raw returncode, captured streams,
truncation marks, and exactly one attribution — so consumers branch on the
attribution rather than on exception types.

HumanEval runs one batch per candidate function through dr-exec's batch driver
kit: each case is delivered as an incremental NDJSON result the moment it is
produced, so a wall-clock deadline, an output overflow, or a late child death
costs only the unfinished tail — completed cases already survive. `dr-code`
declares its execution budgets and the `OPENBLAS_NUM_THREADS=1` grant at its
call sites, maps dr-exec attributions onto HumanEval case verdicts (for
example a candidate-process crash to a per-case error), and owns the case
schemas, scoring, and caching.

The `process_boundary_only` containment profile provides no operating-system
containment. Candidate code runs as the invoking user with full filesystem,
credential, process, and network reach; a payload writing directly to a file
descriptor can still reach the protocol channel. Run evaluations only on
disposable workers whose permissions, network access, resources, and lifetime
are constrained externally.

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
  --corpus /data/corpus.parquet \
  --run-dir /data/preprocessing/run-id \
  --candidate-evaluation /data/evaluations/run-id \
  --output-dir analysis/preprocessing/run-id

uv run python scripts/compare_preprocessing_runs.py \
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
