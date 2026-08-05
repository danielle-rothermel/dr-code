"""Extract the ``[[ ## code ## ]]`` field-marker value as a candidate set."""

from __future__ import annotations

from typing import ClassVar

from dr_code.humaneval.code_parsing import field_marker_value
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import (
    FailureCode,
    Step,
    StepFailedError,
    StepOutput,
    StepSettings,
)
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CandidateOrigin,
    CodeCandidate,
    CodeCandidateSetArtifact,
    ExtractionOperation,
    TextArtifact,
)


class FieldMarkerExtractSettings(StepSettings):
    """The field name whose ``[[ ## name ## ]]`` marker delimits the code."""

    field_name: str = "code"


class FieldMarkerExtract(Step[FieldMarkerExtractSettings]):
    """Text -> CandidateSet for the strict field-marker profile.

    Wraps ``code_parsing.field_marker_value``: extracts the text between
    the matching field marker and the next marker (or end). The value is
    stripped; a missing or empty marker is a data failure
    (``StepFailedError``), not a guess.
    """

    NAME: ClassVar[StepName] = StepName.FIELD_MARKER_EXTRACT
    VERSION: ClassVar[str] = "0"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    Settings = FieldMarkerExtractSettings

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, TextArtifact)
        field_value = field_marker_value(
            value.text, field_name=self.settings.field_name
        )
        if field_value is None:
            raise StepFailedError(
                FailureCode.MISSING_FIELD_MARKER,
                f"missing field marker for {self.settings.field_name!r}",
            )
        source = field_value.strip()
        if not source:
            raise StepFailedError(
                FailureCode.EMPTY_FIELD_MARKER_VALUE,
                "empty field-marker code",
            )
        # The marker value is the whole extracted region, so its origin is
        # this step's operation at the input's single location.
        candidate = CodeCandidate(
            source=source,
            origins=(
                CandidateOrigin(
                    operation=ExtractionOperation(
                        operation_name=self.NAME.value
                    ),
                    input_location=0,
                ),
            ),
        )
        return StepOutput(
            value=CodeCandidateSetArtifact(candidates=(candidate,)),
            facts={"field_name": self.settings.field_name},
        )


__all__ = ["FieldMarkerExtract", "FieldMarkerExtractSettings"]
