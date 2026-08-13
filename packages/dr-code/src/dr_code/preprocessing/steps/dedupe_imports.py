from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.import_inference import dedupe_import_lines
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import CandidateMapStep


class DedupeImports(CandidateMapStep):
    NAME: ClassVar[StepName] = StepName.DEDUPE_IMPORTS
    VERSION: ClassVar[str] = "0"

    def apply_to_candidate(self, source: str) -> str:
        return dedupe_import_lines(source)


__all__ = ["DedupeImports"]
