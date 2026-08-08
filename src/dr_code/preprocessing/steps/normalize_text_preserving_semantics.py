from __future__ import annotations

import ast
import unicodedata
from typing import ClassVar

from dr_code.core.source.python_analysis import (
    validate_python_source_with_ast,
)
from dr_code.core.source.text_transforms import (
    collapse_blank_runs,
    normalize_line_endings,
    normalize_text,
    strip_trailing_whitespace,
)
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import CandidateMapStep


class NormalizeTextPreservingSemantics(CandidateMapStep):
    NAME: ClassVar[StepName] = StepName.NORMALIZE_TEXT_PRESERVING_SEMANTICS
    VERSION: ClassVar[str] = "0"

    def apply_to_candidate(self, source: str) -> str:
        original_result = validate_python_source_with_ast(source)
        if not original_result.validation.compile_ok:
            return normalize_text(source)

        normalized = normalize_line_endings(source)
        normalized = unicodedata.normalize("NFKC", normalized)
        normalized = strip_trailing_whitespace(normalized)
        normalized = collapse_blank_runs(normalized).strip("\n")

        normalized_result = validate_python_source_with_ast(normalized)
        original_tree = original_result.tree
        normalized_tree = normalized_result.tree
        if (
            normalized_result.validation.compile_ok
            and original_tree is not None
            and normalized_tree is not None
            and ast.dump(original_tree) == ast.dump(normalized_tree)
        ):
            return normalized

        return source.strip("\n")


__all__ = ["NormalizeTextPreservingSemantics"]
