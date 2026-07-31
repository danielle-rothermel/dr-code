"""Reject blank decoder output before candidate extraction."""

from __future__ import annotations

from typing import ClassVar

from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import Step, StepFailedError, StepOutput
from dr_code.trace import Artifact, ArtifactKind, TextArtifact


class RequireNonblankText(Step):
    """Pass nonblank text through unchanged; blank decoder output is absent."""

    NAME: ClassVar[StepName] = StepName.REQUIRE_NONBLANK_TEXT
    VERSION: ClassVar[str] = "1"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, TextArtifact)
        facts = {
            "text_character_count": len(value.text),
            "is_nonblank": bool(value.text.strip()),
        }
        if not value.text.strip():
            raise StepFailedError(
                "decoder output is blank",
                failure_code=PreprocessingFailureCode.DECODER_OUTPUT_BLANK,
                facts=facts,
            )
        return StepOutput(value=value, facts=facts)


__all__ = ["RequireNonblankText"]
