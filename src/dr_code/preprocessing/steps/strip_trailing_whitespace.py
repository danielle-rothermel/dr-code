"""Strip trailing whitespace from every line."""

from __future__ import annotations

from typing import ClassVar

from dr_code.core.source.text_transforms import strip_trailing_whitespace
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepOutput
from dr_code.trace import Artifact, ArtifactKind, TextArtifact


class StripTrailingWhitespace(Step):
    """Strip trailing whitespace from every LF-separated line."""

    NAME: ClassVar[StepName] = StepName.STRIP_TRAILING_WHITESPACE
    VERSION: ClassVar[str] = "0"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, TextArtifact)
        return StepOutput(
            value=TextArtifact(text=strip_trailing_whitespace(value.text))
        )


__all__ = ["StripTrailingWhitespace"]
