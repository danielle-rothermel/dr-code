"""Replace Unicode smart quotes with ASCII counterparts."""

from __future__ import annotations

from typing import ClassVar

from dr_code.text_transforms import normalize_smart_quotes
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepOutput
from dr_code.trace import ArtifactKind, TextArtifact


class NormalizeSmartQuotes(Step):
    """Replace Unicode "smart" quotes with their ASCII counterparts."""

    NAME: ClassVar[StepName] = StepName.NORMALIZE_SMART_QUOTES
    VERSION: ClassVar[str] = "1"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT

    def apply(self, value: TextArtifact) -> StepOutput:
        return StepOutput(
            value=TextArtifact(text=normalize_smart_quotes(value.text))
        )


__all__ = ["NormalizeSmartQuotes"]
