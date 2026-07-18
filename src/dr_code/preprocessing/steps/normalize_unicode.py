"""NFKC Unicode normalization."""

from __future__ import annotations

import unicodedata
from typing import ClassVar

from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepOutput
from dr_code.trace import Artifact, ArtifactKind, TextArtifact


class NormalizeUnicode(Step):
    """Apply NFKC Unicode normalization.

    Lifted out of ``normalize_text`` as its own atomic step.
    """

    NAME: ClassVar[StepName] = StepName.NORMALIZE_UNICODE
    VERSION: ClassVar[str] = "1"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, TextArtifact)
        return StepOutput(
            value=TextArtifact(text=unicodedata.normalize("NFKC", value.text))
        )


__all__ = ["NormalizeUnicode"]
