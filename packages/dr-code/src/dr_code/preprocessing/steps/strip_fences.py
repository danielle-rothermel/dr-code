from __future__ import annotations

from typing import ClassVar

from dr_code.core.source.text_transforms import strip_code_fences
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import CandidateMapStep


class StripFences(CandidateMapStep):
    NAME: ClassVar[StepName] = StepName.STRIP_FENCES
    VERSION: ClassVar[str] = "0"

    def apply_to_candidate(self, source: str) -> str:
        return strip_code_fences(source)


__all__ = ["StripFences"]
