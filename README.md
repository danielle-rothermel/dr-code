# dr-code

`dr-code` is a Python code-evaluation toolkit for HumanEval+ execution,
declared preprocessing pipelines, producer-blind metrics, and deterministic
synthetic corruption datasets.

The package provides four current boundaries:

- `dr_code.humaneval` applies its acceptance policy to the preprocessing
  candidate set, resolves the current HumanEval profiles, and evaluates
  accepted code in the configured OCI sandbox.
- `dr_code.preprocessing` binds registered, named steps into ordered
  definitions and emits typed traces with step artifacts, causal absences,
  facts, and producer coordinates.
- `dr_code.metrics` evaluates declared questions against traces and emits
  records containing operator, producer, and metrics-definition coordinates.
- `dr_code.synthetic` builds deterministic corruption datasets from explicit
  recipes.

## Component versions

Semantic provenance uses explicit component coordinates: a stable component
name or definition id, its manually maintained version, and the relevant
ordered composition and settings. Hashes of source code or serialized
definitions are not semantic identities.

Every production preprocessing step, metric operator, preprocessing
definition, scoring profile, metrics profile, HumanEval
override set, synthetic corruption, and synthetic recipe currently has version
`"0"`. This is the unreleased component contract. The repository configuration
records that state for tooling:

```toml
[tool.dr-code.component-versioning]
development-mode = true
initial-version = "0"
```

The marker is read only by repository contract tests. Production code does
not import it, branch on it, or write it into traces, records, cache inputs, or
other runtime artifacts. Package versions, schema versions, dependency
versions, dataset revisions, and byte-integrity digests are separate
contracts.

Development mode ends at the first tagged release, at which point the marker
is removed and real per-component versioning begins.

After development mode ends, component versions remain `"0"` until that
specific component's semantics change. Bump the owning component when:

- a preprocessing step changes accepted inputs, outputs, facts, defaults, or
  failure behavior;
- a metric operator changes facts, execution requests, defaults,
  applicability, or failure attribution;
- a definition or profile changes ordering, composition, bindings, or default
  settings; or
- another semantic stage changes its observable inputs, outputs, policy, or
  failure behavior.

Comments, formatting, types, and demonstrated behavior-preserving refactors
do not require a component-version bump.

Preprocessing trace producers persist the complete definition coordinate:
definition id and version, then each ordered step instance with its registered
step name, independently declared step version, and explicit immutable setting
entries. Registered preprocessing execution resolves the canonical definition
and rejects caller-built objects that claim its coordinate. Unregistered
pipelines use a distinct external-preprocessing producer variant; unrelated
external traces use the external producer variant. Metric records nest the
trace producer coordinate and the complete ordered metrics definition instead
of flattening or abbreviating either composite coordinate.

HumanEval snapshots persist the registered override set as an explicit id,
version, and ordered concrete entries, including typed source-replacement
entries. Synthetic samples nest a structured semantic coordinate containing
the HumanEval task id, generation seed, and complete recipe coordinate. The
plain `sample_id` is only a concise display label.

Cryptographic digests are used only for the immutable OCI image coordinate and
private execution-cache mechanics. The cache key is not a public API or
provenance coordinate, and tests prove reuse and alias resistance through
observable outcomes and call counts. Synthetic RNG initialization supplies the
stable serialization of the structured sample coordinate directly to Python's
product-owned random generator.

## Development

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -q
```

Node-based viewer packages use the version in `.nvmrc` and the lockfile under
`viewer/`.
