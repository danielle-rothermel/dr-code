# Seeded behavioral-mutant suite for HumanEval+

A contamination instrument: rule-based, **seeded**, **behavior-altering**
mutations of the canonical HumanEval+ ground-truth solutions, each validated by
an **execution-derived oracle** over the task's own HumanEval+ test inputs. The
suite is HumanEval+-loader-compatible so downstream (enc-dec) consumption is
trivial.

This fills a gap the literature verifiably lacks: the closest prior work is
arXiv:2503.02296 (stochastic LLM rewrites + execution-derived oracles +
Memorization Risk Index) and EvoEval-Subtle (behavior-altering, static/LLM), but
nothing is simultaneously *seeded* + *behavior-altering* + *execution-oracle* +
*HumanEval+-compatible*.

## Pipeline

For each `(task, operator family, seed)`:

1. Enumerate applicable **sites** in a stable pre-order over the AST of the full
   ground-truth program (`prompt + canonical_solution`, which is what parses).
2. Select a site by a seeded permutation derived from
   `sha256(task_id | family | seed | site_index)`, so `mutant = f(task_id,
   family, seed)` is deterministic.
3. Apply the one-site mutation and `ast.unparse` the result.
4. Derive the oracle: run the mutant on the task's existing test inputs in the
   isolated subprocess, capturing each output `repr` (or the exception type).
5. Apply the validation gates (below). On rejection, advance deterministically
   to the next candidate site; if none is accepted, record a skip with a reason.

Determinism: the only randomness is the seeded site permutation. Given a pinned
config the output is byte-identical (asserted by regeneration goldens).

## Operator families (preliminary set)

| family | edit | notes |
| --- | --- | --- |
| `comparison_flip` | `<`↔`<=`, `>`↔`>=` | strictness flips (boundary behavior); `==`/`!=` deliberately excluded |
| `boundary_shift` | `+1` on an int literal in a comparison / slice / `range()` | classic off-by-one; booleans excluded (they subclass `int`) |
| `aggregation_swap` | `min`↔`max` | always type-safe as a name swap; `sum->len` excluded (context-dependent type safety) |
| `branch_swap` | swap the two arms of an `if` with both a body and an `else` | elif chains excluded (non-local control-flow reshaping) |
| `range_inclusivity` | reduce the `stop` arg of `range()` by 1 | inclusive/exclusive loop-extent bug |

## Validation gates (per mutant)

A mutant is accepted only if it clears every gate:

- **terminates** within the timeout on all inputs (else `OracleError` → try next);
- **deterministic** across two independent runs (identical outputs);
- **behaviorally distinct** from the canonical solution on ≥1 test input;
- **serializable/comparable** outputs (`repr` round-trips; exceptions captured as
  the exception type name, so a mutant that raises on some inputs is still
  comparable rather than crashing the batch).

The search over sites/seeds when a mutant is behaviorally silent is recorded in
the manifest `skipped` list with a reason.

## Artifact schema

Two files at the output directory:

- `mutants.jsonl` — one `MutantRecord` per accepted mutant (sorted by
  `(task_id, family, seed)`, sorted JSON keys, trailing newline per line).
- `manifest.json` — the `MutantManifest` (config + config identity + per-family
  accepted counts + the skip/search log).

`MutantRecord` fields (loader-compatible: `task_id`, `entry_point`, `prompt`,
plus a full mutated program that defines `entry_point`):

- `task_id`, `entry_point`, `prompt`
- `canonical_full_source`, `mutated_full_source` (normalized via `ast.unparse`)
- `operator_family`, `seed`, `site_node_path`, `site_description` (provenance)
- `input_reprs` — the exact test inputs the oracle ran on
- `mutant_expected`, `canonical_expected` — per-input `{kind, output_repr}`
- `distinct_input_indices` — indices where the mutant diverges from canonical
- `diff_summary` — unified diff canonical→mutant
- `canonical_test` — cross-reference to the canonical suite (needed downstream
  for dual/attractor-pull scoring)
- `optional_identifier_rename` — set when `--compose-rename` renamed the entry
  point (behavior-preserving; see below)

`GenerationConfig` (identity-bearing, schema `dr_code.mutants.generation_config`):
`generator_version`, `dataset_schema_version`, `dataset_id`, `hf_revision`,
`operator_families`, `seeds`, `max_inputs_per_mutant`, `timeout_seconds`,
`compose_rename`, `task_filter`. Its SHA-256 identity is the `config_identity`.

## Optional composed identifier rename (flag)

`--compose-rename` additionally renames each mutant's entry-point function to
`target_fxn` with all-occurrence coverage (definition + recursive self-calls).
This is **behavior-preserving** (a pure rename), so the oracle expected outputs
are unchanged; it only breaks surface-level name memorization. It is opt-in and
part of the config identity.

## CLI

```
# List applicable sites without executing or writing:
python -m dr_code.mutants generate --dry-run --tasks HumanEval/0 --operators all

# Deterministic generation (one command; regeneration is byte-identical):
python -m dr_code.mutants generate \
    --output-dir <dir> --operators all --seeds 1 --max-inputs 50
```

Flags: `--operators` (comma list or `all`), `--seeds N`, `--tasks` (comma list
or empty for all 164), `--max-inputs`, `--timeout`, `--snapshot/--hf`,
`--compose-rename`, `--dry-run`.

## Known limitations / publication-hardening TODOs

Preliminary-results scope. Flagged for later hardening rather than solved now:

- **Operator coverage breadth.** Only 5 families; add arithmetic-operator swaps,
  logical-connective swaps (`and`↔`or`), return-constant replacement, and
  augmented-assignment flips. `boundary_shift` currently shifts only a direct int
  literal (not `len(x) - 1`-style `BinOp` bounds) and only by `+1`.
- **Mutant difficulty calibration.** Distinctness is a floor, not a difficulty
  measure; we do not yet stratify by how many inputs distinguish, or by semantic
  subtlety. `max_inputs_per_mutant` is a deterministic *head* subsample — a
  coverage- or difficulty-aware input selection would be stronger.
- **Spec/docstring regeneration for direct-generation arms.** We deliberately do
  NOT regenerate the prompt/docstring/spec to match the mutated behavior, because
  the enc-dec consumer needs only the mutated *code body*. Direct-generation
  arms (prompt→code) would need a regenerated spec; out of scope here.
- **Single-site mutations only.** No higher-order (multi-site) mutants yet.
- **Oracle host isolation, not security.** The subprocess is dr-code's
  `run_python_subprocess` (host user permissions); run on a disposable host.
