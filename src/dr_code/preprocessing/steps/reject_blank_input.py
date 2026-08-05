"""Reject blank input explicitly, before any extraction is attempted."""

from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import (
    Step,
    StepFailedError,
    StepOutput,
    StepSettings,
)
from dr_code.trace import Artifact, ArtifactKind, TextArtifact


class RejectBlankInput(Step[StepSettings]):
    """Pass text through unchanged unless it is blank.

    Empty or whitespace-only input names its own failure kind rather than
    surfacing later as "nothing extracted": there was never anything to
    extract. The distinction is what lets a consumer tell an empty response
    apart from a response that carried no recoverable code.
    """

    NAME: ClassVar[StepName] = StepName.REJECT_BLANK_INPUT
    VERSION: ClassVar[str] = "0"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, TextArtifact)
        if not value.text.strip():
            raise StepFailedError(
                PreprocessingFailureCode.BLANK_INPUT,
                "input text is empty or whitespace-only",
                evidence={"input_length": len(value.text)},
            )
        return StepOutput(value=value)


__all__ = ["RejectBlankInput"]
