"""Strip code fences from each candidate."""

from __future__ import annotations

from typing import ClassVar

from dr_code.text_transforms import strip_code_fences
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import CandidateMapStep


class StripFences(CandidateMapStep):
    """Drop a leading and/or trailing fence line from each candidate."""

    NAME: ClassVar[StepName] = StepName.STRIP_FENCES
    VERSION: ClassVar[str] = "0"

    def apply_to_candidate(self, source: str) -> str:
        return strip_code_fences(source)


__all__ = ["StripFences"]
