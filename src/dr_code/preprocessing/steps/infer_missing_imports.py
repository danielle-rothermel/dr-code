"""Prepend inferred missing imports per candidate."""

from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.import_inference import infer_missing_imports
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import CandidateMapStep


class InferMissingImports(CandidateMapStep):
    """Prepend inferred missing imports to each candidate.

    Wraps ``import_inference.infer_missing_imports`` — the second
    constituent of ``infer_necessary_imports``.
    """

    NAME: ClassVar[StepName] = StepName.INFER_MISSING_IMPORTS
    VERSION: ClassVar[str] = "1"

    def apply_to_candidate(self, source: str) -> str:
        return infer_missing_imports(source)


__all__ = ["InferMissingImports"]
