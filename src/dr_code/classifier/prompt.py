"""Prompt construction for the failure classifier."""

from __future__ import annotations

from typing import Final

from dr_code.classifier.taxonomy import (
    FailureKind,
    definitions_block,
    label_names,
)


_MAX_TEXT_CHARS: Final = 6000

_KIND_NOUN: Final = {
    FailureKind.PARSE: (
        "a preprocessing PARSE/EXTRACTION failure: a language model emitted "
        "nonblank text but the pipeline extracted no usable Python "
        "top-level-function candidate"
    ),
    FailureKind.TEST: (
        "a TEST failure: a candidate compiled but did not pass its evaluation"
    ),
}


def build_prompt(kind: FailureKind, text: str) -> str:
    """Build the classification prompt for one failure item."""
    labels = ", ".join(label_names(kind))
    body = text if len(text) <= _MAX_TEXT_CHARS else (
        text[:_MAX_TEXT_CHARS] + "\n...[truncated for classification]"
    )
    return (
        "You are classifying "
        f"{_KIND_NOUN[kind]}.\n\n"
        "Choose exactly one label from this taxonomy:\n"
        f"{definitions_block(kind)}\n\n"
        "Valid labels (use one verbatim): "
        f"{labels}\n\n"
        "The failing output to classify is between the markers:\n"
        "<<<FAILURE\n"
        f"{body}\n"
        "FAILURE>>>\n\n"
        'Reply with ONLY a JSON object: {"label": "<one label>", '
        '"rationale": "<one concise line explaining the choice>"}. '
        "No prose, no code fences, no extra keys."
    )


__all__ = ("build_prompt",)
