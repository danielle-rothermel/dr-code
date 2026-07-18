"""Extract the ``[[ ## code ## ]]`` field-marker value as a candidate set."""

from __future__ import annotations

from typing import ClassVar

from dr_code.humaneval.code_parsing import field_marker_value
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import (
    Step,
    StepFailedError,
    StepOutput,
    StepSettings,
)
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CodeCandidateSetArtifact,
    TextArtifact,
)


class FieldMarkerSettings(StepSettings):
    """The field name whose ``[[ ## name ## ]]`` marker delimits the code."""

    field_name: str = "code"


class FieldMarker(Step[FieldMarkerSettings]):
    """Text -> CandidateSet for the strict field-marker profile.

    Wraps ``code_parsing.field_marker_value``: extracts the text between
    the matching field marker and the next marker (or end). The value is
    stripped; a missing or empty marker is a data failure
    (``StepFailedError``), not a guess.
    """

    NAME: ClassVar[StepName] = StepName.FIELD_MARKER_EXTRACT
    VERSION: ClassVar[str] = "1"
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.TEXT
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    Settings = FieldMarkerSettings

    def apply(self, value: Artifact) -> StepOutput:
        assert isinstance(value, TextArtifact)
        field_value = field_marker_value(
            value.text, field_name=self.settings.field_name
        )
        if field_value is None:
            raise StepFailedError(
                f"missing field marker for {self.settings.field_name!r}"
            )
        candidate = field_value.strip()
        if not candidate:
            raise StepFailedError("empty field-marker code")
        return StepOutput(
            value=CodeCandidateSetArtifact(candidates=(candidate,)),
            facts={"field_name": self.settings.field_name},
        )


__all__ = ["FieldMarker", "FieldMarkerSettings"]
