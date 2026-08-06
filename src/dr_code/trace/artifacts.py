from __future__ import annotations

import ast
import math
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, field_validator, model_validator

from dr_code.core.models import FrozenModel


class ArtifactKind(StrEnum):
    TEXT = "text"
    CODE = "code"
    CODE_CANDIDATE_SET = "code_candidate_set"
    INSPECTED_CODE_CANDIDATE_SET = "inspected_code_candidate_set"
    JSON = "json"


class TextArtifact(FrozenModel):
    kind: Literal[ArtifactKind.TEXT] = ArtifactKind.TEXT
    text: str


class CodeArtifact(FrozenModel):
    """Python source with no parse or compile guarantee."""

    kind: Literal[ArtifactKind.CODE] = ArtifactKind.CODE
    source: str


class ExtractionOperation(FrozenModel):
    operation_name: str


class CandidateOrigin(FrozenModel):
    operation: ExtractionOperation
    input_location: int = Field(ge=0)


class CodeCandidate(FrozenModel):
    """Origins are non-empty and preserve transform/deduplication encounter order."""

    source: str
    origins: tuple[CandidateOrigin, ...] = Field(min_length=1)

    def extended(
        self, origin: CandidateOrigin, *, source: str
    ) -> CodeCandidate:
        return CodeCandidate(source=source, origins=(*self.origins, origin))


class CodeCandidateSetArtifact(FrozenModel):
    kind: Literal[ArtifactKind.CODE_CANDIDATE_SET] = (
        ArtifactKind.CODE_CANDIDATE_SET
    )
    candidates: tuple[CodeCandidate, ...]


class CandidateInspection(FrozenModel):
    parses: bool
    parse_error: str | None = None
    compiles: bool
    compile_error: str | None = None
    top_level_function_names: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_structural_consistency(self) -> Self:
        if self.compiles and not self.parses:
            raise ValueError("source that does not parse cannot compile")
        if self.parses != (self.parse_error is None):
            raise ValueError(
                "parse_error must be present exactly when parses is False"
            )
        if self.compiles != (self.compile_error is None):
            raise ValueError(
                "compile_error must be present exactly when compiles is False"
            )
        return self


class InspectedCodeCandidate(FrozenModel):
    candidate: CodeCandidate
    inspection: CandidateInspection


class InspectedCodeCandidateSetArtifact(FrozenModel):
    kind: Literal[ArtifactKind.INSPECTED_CODE_CANDIDATE_SET] = (
        ArtifactKind.INSPECTED_CODE_CANDIDATE_SET
    )
    candidates: tuple[InspectedCodeCandidate, ...]


class JsonArtifact(FrozenModel):
    """JSON payload restricted to finite numbers."""

    kind: Literal[ArtifactKind.JSON] = ArtifactKind.JSON
    payload: JsonValue

    @field_validator("payload")
    @classmethod
    def reject_non_finite_floats(cls, payload: JsonValue) -> JsonValue:
        if _contains_non_finite_float(payload):
            raise ValueError(
                "JSON artifact payload must contain only finite floats"
            )
        return payload


def _contains_non_finite_float(value: JsonValue) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, list):
        return any(_contains_non_finite_float(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_non_finite_float(item) for item in value.values())
    return False


Artifact = Annotated[
    TextArtifact
    | CodeArtifact
    | CodeCandidateSetArtifact
    | InspectedCodeCandidateSetArtifact
    | JsonArtifact,
    Field(discriminator="kind"),
]


def parsed_module(code: CodeArtifact) -> ast.Module:
    return ast.parse(code.source)


__all__ = [
    "Artifact",
    "ArtifactKind",
    "CandidateInspection",
    "CandidateOrigin",
    "CodeArtifact",
    "CodeCandidate",
    "CodeCandidateSetArtifact",
    "ExtractionOperation",
    "InspectedCodeCandidate",
    "InspectedCodeCandidateSetArtifact",
    "JsonArtifact",
    "TextArtifact",
    "parsed_module",
]
