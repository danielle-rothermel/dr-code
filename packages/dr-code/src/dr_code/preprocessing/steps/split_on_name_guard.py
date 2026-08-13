from __future__ import annotations

from typing import ClassVar

from dr_code.core.source.text_transforms import drop_if_name
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import CandidateMapStep


class SplitOnNameGuard(CandidateMapStep):
    NAME: ClassVar[StepName] = StepName.SPLIT_ON_NAME_GUARD
    VERSION: ClassVar[str] = "0"

    def apply_to_candidate(self, source: str) -> list[str]:
        return drop_if_name(source)


__all__ = ["SplitOnNameGuard"]
