"""Deduplicate import lines per candidate."""

from __future__ import annotations

from typing import ClassVar

from dr_code.humaneval.import_inference import _dedup_imports
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import CandidateMapStep


class DedupeImports(CandidateMapStep):
    """Remove duplicate import lines from each candidate.

    Wraps ``import_inference._dedup_imports`` — the third constituent of
    ``infer_necessary_imports``.
    """

    NAME: ClassVar[StepName] = StepName.DEDUPE_IMPORTS
    VERSION: ClassVar[str] = "1"

    def apply_to_candidate(self, source: str) -> str:
        return _dedup_imports(source)


__all__ = ["DedupeImports"]
