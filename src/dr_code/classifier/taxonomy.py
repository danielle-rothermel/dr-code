"""Closed, versioned failure taxonomies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from dr_code.eval.identity import identity_hash_for

TAXONOMY_VERSION: Final = "failure-taxonomy-v1"
OTHER_LABEL: Final = "other"


class FailureFamily(StrEnum):
    """Independent failure families classified by this package."""

    PARSE = "parse"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class TaxonomyLabel:
    name: str
    definition: str


_PARSE_LABELS: Final = (
    TaxonomyLabel(
        "truncated-output",
        "Code ends abruptly mid-token, statement, expression, or function.",
    ),
    TaxonomyLabel(
        "markdown-fence-mangling",
        "Markup or response scaffolding prevents extraction of a code body.",
    ),
    TaxonomyLabel(
        "prose-no-code",
        "Natural-language prose contains no self-contained function definition.",
    ),
    TaxonomyLabel(
        "json-wrapped-code",
        "A structured envelope carries nonempty code instead of bare Python.",
    ),
    TaxonomyLabel(
        "expression-not-function",
        "Python is only an expression, comprehension, or assignment.",
    ),
    TaxonomyLabel(
        "script-not-function",
        "Python is a top-level script rather than a reusable function.",
    ),
    TaxonomyLabel(
        "syntax-error-other",
        "A function is attempted but has a syntax error other than truncation.",
    ),
    TaxonomyLabel(
        "empty-code",
        "A response wrapper or code block has an explicitly empty payload.",
    ),
    TaxonomyLabel(
        "non-python",
        "The response is another language or pseudocode.",
    ),
    TaxonomyLabel(
        OTHER_LABEL,
        "The parse failure does not fit another label.",
    ),
)

_TEST_LABELS: Final = (
    TaxonomyLabel(
        "wrong-edge-case",
        "The main path works but a boundary or special input is mishandled.",
    ),
    TaxonomyLabel(
        "wrong-algorithm",
        "The approach is fundamentally incorrect for the task.",
    ),
    TaxonomyLabel(
        "off-by-one",
        "An index, range, or count boundary is wrong.",
    ),
    TaxonomyLabel(
        "type-error",
        "Execution raises a type, attribute, value, or conversion error.",
    ),
    TaxonomyLabel(
        "timeout",
        "Evaluation exceeds its time budget.",
    ),
    TaxonomyLabel(
        OTHER_LABEL,
        "The test failure does not fit another label.",
    ),
)

_LABELS: Final = {
    FailureFamily.PARSE: _PARSE_LABELS,
    FailureFamily.TEST: _TEST_LABELS,
}


def labels_for(family: FailureFamily) -> tuple[TaxonomyLabel, ...]:
    return _LABELS[family]


def label_names(family: FailureFamily) -> tuple[str, ...]:
    return tuple(label.name for label in labels_for(family))


def is_valid_label(family: FailureFamily, label: str) -> bool:
    return label in label_names(family)


def definitions_block(family: FailureFamily) -> str:
    return "\n".join(
        f"- {label.name}: {label.definition}" for label in labels_for(family)
    )


def taxonomy_identity() -> str:
    """Hash the exact closed labels and definitions used by the classifier."""
    return identity_hash_for(
        schema="dr_code.failure_classifier.taxonomy",
        payload={
            "taxonomy_version": TAXONOMY_VERSION,
            "families": {
                family.value: [
                    {"name": label.name, "definition": label.definition}
                    for label in labels_for(family)
                ]
                for family in FailureFamily
            },
        },
    )


__all__ = (
    "OTHER_LABEL",
    "TAXONOMY_VERSION",
    "FailureFamily",
    "TaxonomyLabel",
    "definitions_block",
    "is_valid_label",
    "label_names",
    "labels_for",
    "taxonomy_identity",
)
