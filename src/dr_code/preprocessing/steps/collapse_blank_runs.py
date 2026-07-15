"""Collapse runs of three or more newlines to one blank line."""

from __future__ import annotations

from typing import ClassVar

from dr_code.text_transforms import collapse_blank_runs
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepOutput
from dr_code.trace import ArtifactKind, TextArtifact


class CollapseBlankRuns(Step):
    """Collapse runs of three or more newlines down to one blank line."""

    NAME: ClassVar[StepName] = StepName.COLLAPSE_BLANK_RUNS
    VERSION: ClassVar[str] = "1"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT

    def apply(self, value: TextArtifact) -> StepOutput:
        return StepOutput(
            value=TextArtifact(text=collapse_blank_runs(value.text))
        )


__all__ = ["CollapseBlankRuns"]
