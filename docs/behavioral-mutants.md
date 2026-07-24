# Behavioral mutants

`dr_code.mutants` generates deterministic, behavior-altering variants of the
pinned HumanEval+ canonical programs. Each accepted record is tied to one
operator family, seed, concrete syntax-tree site, canonical input sequence,
and execution-derived outcome sequence.

## Operators and search

The generator identity is `mutants@v2`. Its complete operator vocabulary is:

- `comparison_flip`: flips one `<`, `<=`, `>`, or `>=` operator. Each operator
  in a chained comparison is a separate site.
- `boundary_shift`: adds one to one integer comparison operand, slice bound, or
  `range()` argument.
- `aggregation_swap`: replaces one `min()` call with `max()`, or vice versa.
- `branch_swap`: exchanges the arms of one complete `if`/`else`.
- `range_inclusivity`: reduces the stop expression of one `range()` by one.

Sites are addressed by a stable pre-order node index and a target index inside
that node. A SHA-256 ordering derived from task id, operator, seed, and site
index determines the search order. Rejected and duplicate candidates advance
to the next site in that order. Records, skips, and family counts are emitted
in stable order.

## Acceptance

The authenticated HumanEval+ snapshot is an installed package resource and the
offline default. `--hf` independently loads the explicitly pinned Hugging Face
revision without falling back to the snapshot. The selected source is part of
the generation identity and is resolved once. Both the canonical program and a
candidate mutant are executed twice in fresh Python subprocesses. A candidate
is accepted only when both canonical executions agree, both mutant executions
agree, and the mutant differs from the canonical outcome on at least one input.

The generation config names every selected task explicitly. Its identity also
binds a digest of each task id, prompt, entry point, normalized canonical
source, complete canonical test, and exact extracted input representations.
The current subprocess-oracle identity and Python runtime identity are separate
coordinates. Injected test or research runners must supply their own explicit
non-production identities.

Schema-v2 outcomes are a discriminated union. Value outcomes preserve
`value_repr`; error outcomes preserve the exception's module-qualified type and
`exception_args_repr`. This keeps a returned string distinct from an exception
and detects argument changes within the same exception class.

Every execution uses
[`dr_code.execution.subprocess`](../src/dr_code/execution/subprocess.py) with
bounded text input and output, a wall-clock deadline, and process-group
cleanup. This subprocess boundary does not restrict the program's access to
the invoking user's filesystem, processes, credentials, or network. Generate
mutants only from trusted programs on a worker whose permissions and resources
match that risk.

The oracle duplicates its result descriptor before loading a program and
redirects the program's stdout away from that descriptor. The final result is
bound to a fresh invocation id and accepted only as one complete envelope;
program output cannot be parsed as an oracle result.

## Generate

Inspect applicable sites without executing programs or writing files:

```bash
uv run python -m dr_code.mutants generate \
  --dry-run \
  --tasks HumanEval/0 \
  --operators comparison_flip
```

Generate an immutable dataset directory:

```bash
uv run python -m dr_code.mutants generate \
  --output-dir artifacts/mutants/humaneval-plus-v1 \
  --snapshot \
  --seeds 3 \
  --max-inputs 50
```

The destination must not already exist. Generation stages both artifacts in a
private sibling directory and publishes the directory in one atomic,
no-replacement operation:

- `mutants.jsonl` contains current-schema records in stable order.
- `manifest.json` pins the complete generation config, its identity, the
  selected-suite and execution identities, record count, per-family counts,
  full ordered search/skip log, JSONL SHA-256, and resulting dataset identity.

Use `dr_code.mutants.load_dataset()` with a separately trusted
`expected_dataset_identity`, `max_manifest_bytes`, and `max_records_bytes` to
consume a dataset. The two byte limits must come from the caller's trusted
generation or storage policy, never from the artifact's manifest: they bound
unauthenticated capture before parsing or hash verification. Loading is
offline: source content is captured and validated during generation and is not
reacquired after publication. The loader rejects
unknown schemas, config-identity mismatches, record hash or count mismatches,
invalid record identities, duplicate programs, unstable order, incomplete
outcomes, incorrect divergence indices, incomplete or overlapping search
coordinates, unexpected files, manifest identity mismatches, and caller pin
mismatches.
