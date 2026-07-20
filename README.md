# dr-code

## Preprocessing boundary

Preprocessing extracts structurally usable code from text. A definition may
require each returned code candidate to contain a top-level function, but it
does not require that function to have an application- or benchmark-specific
name. Function identity is intentionally outside the repository's generic
preprocessing contract.

Applications that need a particular function name can add an explicit
processing step or filter preprocessing or test results afterward. Executable
tests, rather than name matching in preprocessing, determine whether an
extracted candidate satisfies a task.

The preprocessing trace is the authoritative record of semantic analysis. It
owns stable terminal failure codes, extraction provenance, per-stage candidate
counts, rejection reasons, compilation diagnostics, and structural function
facts. Batch and persistence adapters may attach source/run identity and
mechanically reshape those facts, but should not reclassify preprocessing
results. A missing external value remains an ingestion concern because there
is no text artifact to process.

See the [decoder-output preprocessing analysis plan](docs/decoder-output-preprocessing-plan.html)
for the proposed exhaustive function-candidate pipeline and corpus audit.
