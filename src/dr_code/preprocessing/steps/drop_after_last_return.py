"""Truncate each candidate after its last return line."""

from __future__ import annotations

from typing import ClassVar

from dr_code.text_transforms import drop_after_last_return
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import CandidateMapStep


class DropAfterLastReturn(CandidateMapStep):
    """Truncate each candidate after its last ``return`` line."""

    NAME: ClassVar[StepName] = StepName.DROP_AFTER_LAST_RETURN
    VERSION: ClassVar[str] = "0"

    def apply_to_candidate(self, source: str) -> str:
        return drop_after_last_return(source)


__all__ = ["DropAfterLastReturn"]
