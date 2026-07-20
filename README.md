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

The public HumanEval flow is
`humaneval-function-candidates@v1`. Bind it once for batch work; every
successful output is a nonempty ordered candidate set whose entries compile,
contain at least one top-level function, and carry stable candidate IDs plus
their extraction origins. `[[ ## code ## ]]` is supported as an input
representation inside this flow, not as a separate parser mode.

```python
from dr_code.preprocessing import (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
    bind_preprocessing,
)

runner = bind_preprocessing(
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION
)
```

See the [decoder-output preprocessing analysis plan](docs/decoder-output-preprocessing-plan.html)
for the flow’s design and the reproducible corpus audit built on it.

For full HumanEval+ scoring that requires NumPy, see the
[reproducible sandbox-image build and preflight flow](docs/humaneval-plus-sandbox.md).
