"""Trim leading and trailing newlines."""

from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepOutput
from dr_code.trace import Artifact, ArtifactKind, TextArtifact


class TrimOuterBlanks(Step):
    """Strip a leading/trailing run of newlines.

    Lifted from the final ``text.strip("\\n")`` of ``normalize_text``.
    """

    NAME: ClassVar[StepName] = StepName.TRIM_OUTER_BLANKS
    VERSION: ClassVar[str] = "1"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, TextArtifact)
        return StepOutput(value=TextArtifact(text=value.text.strip("\n")))


__all__ = ["TrimOuterBlanks"]
