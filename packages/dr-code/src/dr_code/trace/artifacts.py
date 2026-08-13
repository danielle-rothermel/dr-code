from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Literal, Self, cast

from pydantic import (
    Field,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)

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
    def validate_and_freeze_payload(cls, payload: JsonValue) -> JsonValue:
        if _contains_non_finite_float(payload):
            raise ValueError(
                "JSON artifact payload must contain only finite floats"
            )
        return freeze_json_value(payload)

    @field_serializer("payload")
    def serialize_payload(self, payload: JsonValue) -> JsonValue:
        return json_value_to_wire(payload)


def _contains_non_finite_float(value: object) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, list | tuple):
        return any(_contains_non_finite_float(item) for item in value)
    if isinstance(value, Mapping):
        return any(_contains_non_finite_float(item) for item in value.values())
    return False


def freeze_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, list):
        return cast(
            JsonValue, tuple(freeze_json_value(item) for item in value)
        )
    if isinstance(value, dict):
        return cast(
            JsonValue,
            MappingProxyType(
                {key: freeze_json_value(item) for key, item in value.items()}
            ),
        )
    return value


def json_value_to_wire(value: object) -> JsonValue:
    if isinstance(value, tuple):
        return [json_value_to_wire(item) for item in value]
    if isinstance(value, Mapping):
        wire: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "immutable JSON mappings must retain string keys"
                )
            wire[key] = json_value_to_wire(item)
        return wire
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(
        f"immutable JSON value has unsupported type: {type(value).__name__}"
    )


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
