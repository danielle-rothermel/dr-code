"""Versioned, seeded failure taxonomy for the LLM classifier.

The taxonomy is a set of constants in code with a ``taxonomy_version`` string.
Parse-failure labels were seeded by studying ~20 real failures from run
``generation-corpus-functions-v1-extraction-redesign-v4-20260722`` (see the
one-line example on each label). Test-failure labels are a smaller separate
seed set. Every label set carries an ``other`` escape hatch, and the classifier
always returns a one-line rationale alongside the label.

Bump ``TAXONOMY_VERSION`` whenever a label's meaning changes; the classifier
keys resumability on it, so a bump forces a re-run of previously labelled items.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


TAXONOMY_VERSION: Final = "failure-taxonomy-v1"

OTHER_LABEL: Final = "other"


class FailureKind(StrEnum):
    """Which failure family a taxonomy applies to."""

    PARSE = "parse"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class TaxonomyLabel:
    """One taxonomy label with a definition and a seed example."""

    name: str
    definition: str
    example: str


# Parse/extraction failures: the decoder emitted nonblank text but the
# preprocessing pipeline extracted no usable top-level-function candidate. Seed
# examples are verbatim (truncated) decoder outputs observed in the pilot run.
_PARSE_LABELS: Final = (
    TaxonomyLabel(
        name="truncated-output",
        definition=(
            "The output is cut off mid-token, mid-statement, or mid-function "
            "so the code cannot be parsed; it ends abruptly rather than at a "
            "natural boundary."
        ),
        example=(
            'def rescale_to_unit(numbers: List[float]) elements), apply a '
            "linear transform ... (garbled, truncated mid-signature)"
        ),
    ),
    TaxonomyLabel(
        name="markdown-fence-mangling",
        definition=(
            "The code is wrapped in chat/markup scaffolding (markdown fences, "
            "[[ ## code ## ]] markers, empty ```python``` blocks) that leaves "
            "no extractable code body."
        ),
        example="[[ ## code ## ]]\n```python\n```\n[[ ## completed ## ]]",
    ),
    TaxonomyLabel(
        name="prose-no-code",
        definition=(
            "The output is natural-language prose, a clarifying question, or "
            "an explanation with no self-contained function definition to "
            "extract."
        ),
        example=(
            'Could you please clarify the exact task? The phrase "..." is '
            "ambiguous ..."
        ),
    ),
    TaxonomyLabel(
        name="json-wrapped-code",
        definition=(
            "The output is a JSON/structured envelope (e.g. {\"code\": ...}) "
            "carrying an expression or snippet rather than a bare Python "
            "function definition."
        ),
        example='{\n  "code": "sum(ord(c) for c in s if c.isupper())"\n}',
    ),
    TaxonomyLabel(
        name="expression-not-function",
        definition=(
            "The output is valid-looking Python but is a bare expression, "
            "comprehension, or assignment with no top-level function "
            "definition."
        ),
        example="[x for x in input_list if type(x) is int]",
    ),
    TaxonomyLabel(
        name="script-not-function",
        definition=(
            "The output is a top-level script (reads stdin, prints results, "
            "or runs statements at module scope) instead of defining a "
            "reusable function."
        ),
        example=(
            "import sys\nfrom collections import Counter\n"
            "text = sys.stdin.read()\n... print(result)"
        ),
    ),
    TaxonomyLabel(
        name="syntax-error-other",
        definition=(
            "The output tries to define a function but contains a syntax "
            "error not explained by truncation (stray tokens, split "
            "identifiers, malformed literals)."
        ),
        example=(
            "def is_non_decreasing_and_at_m most_twice(lst): "
            "(identifier split by a space)"
        ),
    ),
    TaxonomyLabel(
        name="empty-code",
        definition=(
            "The output carries an explicitly empty code payload (e.g. "
            '{"code":""} or an empty fenced block) with nothing to parse.'
        ),
        example='{"code":""}',
    ),
    TaxonomyLabel(
        name="non-python",
        definition=(
            "The code is written in a language other than Python (or is "
            "pseudo-code), so the Python pipeline cannot compile it."
        ),
        example="function add(a, b) { return a + b; }",
    ),
    TaxonomyLabel(
        name=OTHER_LABEL,
        definition=(
            "A nonblank failure that does not fit any other label; use "
            "sparingly and explain in the rationale."
        ),
        example="(escape hatch)",
    ),
)


# Test failures: a candidate compiled but failed evaluation. Seeded as themes
# (no large real test-failure sample was studied; refine from evidence when a
# run with evaluation artifacts is classified).
_TEST_LABELS: Final = (
    TaxonomyLabel(
        name="wrong-edge-case",
        definition=(
            "The solution handles the main path but mishandles a boundary or "
            "special input (empty input, zero, negatives, duplicates)."
        ),
        example="fails only on empty-list / single-element inputs",
    ),
    TaxonomyLabel(
        name="wrong-algorithm",
        definition=(
            "The approach is fundamentally incorrect for the task, failing "
            "broadly rather than on a narrow edge case."
        ),
        example="returns the sum where the task asked for the product",
    ),
    TaxonomyLabel(
        name="off-by-one",
        definition=(
            "The logic is nearly right but for an index/range/count boundary "
            "error (inclusive vs exclusive, len vs len-1)."
        ),
        example="range(1, n) where range(1, n + 1) was needed",
    ),
    TaxonomyLabel(
        name="type-error",
        definition=(
            "Execution raises a type/attribute/value error at runtime (wrong "
            "types, None handling, bad conversion)."
        ),
        example="TypeError: unsupported operand type(s) for +",
    ),
    TaxonomyLabel(
        name="timeout",
        definition=(
            "The candidate does not finish within the evaluation time budget "
            "(infinite loop or too-slow algorithm)."
        ),
        example="evaluation timed out after the per-case budget",
    ),
    TaxonomyLabel(
        name=OTHER_LABEL,
        definition=(
            "A test failure that does not fit any other label; use sparingly "
            "and explain in the rationale."
        ),
        example="(escape hatch)",
    ),
)


_LABELS_BY_KIND: Final[dict[FailureKind, tuple[TaxonomyLabel, ...]]] = {
    FailureKind.PARSE: _PARSE_LABELS,
    FailureKind.TEST: _TEST_LABELS,
}


def labels_for(kind: FailureKind) -> tuple[TaxonomyLabel, ...]:
    """Return the ordered taxonomy labels for one failure kind."""
    return _LABELS_BY_KIND[kind]


def label_names(kind: FailureKind) -> tuple[str, ...]:
    """Return the valid label names for one failure kind, in order."""
    return tuple(label.name for label in _LABELS_BY_KIND[kind])


def is_valid_label(kind: FailureKind, name: str) -> bool:
    """Report whether ``name`` is a defined label for ``kind``."""
    return name in label_names(kind)


def definitions_block(kind: FailureKind) -> str:
    """Render the taxonomy definitions as a numbered prompt block."""
    lines = []
    for label in _LABELS_BY_KIND[kind]:
        lines.append(f"- {label.name}: {label.definition}")
    return "\n".join(lines)


__all__ = (
    "OTHER_LABEL",
    "TAXONOMY_VERSION",
    "FailureKind",
    "TaxonomyLabel",
    "definitions_block",
    "is_valid_label",
    "label_names",
    "labels_for",
)
