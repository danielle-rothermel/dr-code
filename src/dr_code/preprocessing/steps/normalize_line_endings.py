"""Normalize CRLF/CR line endings to LF."""

from __future__ import annotations

from typing import ClassVar

from dr_code.text_transforms import normalize_line_endings
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepOutput
from dr_code.trace import Artifact, ArtifactKind, TextArtifact


class NormalizeLineEndings(Step):
    """Convert CRLF and bare CR line endings to LF."""

    NAME: ClassVar[StepName] = StepName.NORMALIZE_LINE_ENDINGS
    VERSION: ClassVar[str] = "1"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, TextArtifact)
        return StepOutput(
            value=TextArtifact(text=normalize_line_endings(value.text))
        )


__all__ = ["NormalizeLineEndings"]
