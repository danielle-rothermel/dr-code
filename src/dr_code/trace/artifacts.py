"""Artifact discriminated union + derived-view helpers."""

from __future__ import annotations

import ast
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from dr_code.models import FrozenModel


class ArtifactKind(StrEnum):
    TEXT = "text"
    CODE = "code"
    CODE_CANDIDATE_SET = "code_candidate_set"
    IDENTIFIED_CANDIDATE_SET = "identified_candidate_set"
    JSON = "json"


class TextArtifact(FrozenModel):
    kind: Literal[ArtifactKind.TEXT] = ArtifactKind.TEXT
    text: str


class CodeArtifact(FrozenModel):
    """Source text that has passed a compile check upstream.

    Canonical value is the source string only; the AST is a derived
    view (S3).
    """

    kind: Literal[ArtifactKind.CODE] = ArtifactKind.CODE
    source: str


class ExtractionOperation(FrozenModel):
    """One ordered operation that contributed to a candidate source."""

    kind: str
    details: dict[str, JsonValue] = {}


class CandidateOrigin(FrozenModel):
    """One complete ordered path that yielded a candidate source."""

    path: tuple[ExtractionOperation, ...] = Field(min_length=1)


class CandidateLineage(FrozenModel):
    """Stable post-cleaning identity plus every extraction origin."""

    candidate_id: str | None = None
    origins: tuple[CandidateOrigin, ...] = Field(min_length=1)


class CodeCandidateSetArtifact(FrozenModel):
    """Ordered candidate sources, conservative first. Fan-out as data
    (P-S2)."""

    kind: Literal[ArtifactKind.CODE_CANDIDATE_SET] = (
        ArtifactKind.CODE_CANDIDATE_SET
    )
    candidates: tuple[str, ...]
    lineage: tuple[CandidateLineage, ...]

    @model_validator(mode="after")
    def _validate_lineage_alignment(self) -> CodeCandidateSetArtifact:
        if len(self.lineage) != len(self.candidates):
            raise ValueError(
                "candidate lineage must be aligned with candidates"
            )
        return self


class CandidateInspection(FrozenModel):
    """Serializable Python facts derived by one parse/compile inspection."""

    parse_ok: bool
    parse_error: str | None
    compile_ok: bool
    compile_error: str | None
    compile_warnings: tuple[str, ...] = ()
    parser_stack_overflow: bool = False
    parser_recursion_overflow: bool = False
    is_plain_literal_module: bool = False
    is_code_repr_assignment: bool = False
    top_level_function_names: tuple[str, ...] = ()
    top_level_async_function_names: tuple[str, ...] = ()


class IdentifiedCandidate(FrozenModel):
    """One final cleaned source with identity, provenance, and inspection."""

    source: str
    lineage: CandidateLineage
    inspection: CandidateInspection

    @model_validator(mode="after")
    def _validate_candidate_identity(self) -> IdentifiedCandidate:
        if self.lineage.candidate_id is None:
            raise ValueError("identified candidate requires candidate_id")
        return self


class IdentifiedCandidateSetArtifact(FrozenModel):
    """Internal parse-once candidate representation used by policy steps."""

    kind: Literal[ArtifactKind.IDENTIFIED_CANDIDATE_SET] = (
        ArtifactKind.IDENTIFIED_CANDIDATE_SET
    )
    candidates: tuple[IdentifiedCandidate, ...]


class JsonArtifact(FrozenModel):
    """Escape hatch for externally built traces (X-S2), e.g. a HumanEval
    task payload for code_test. Consumers revalidate `payload` into their
    own model at bind time; a payload that fails validation is a
    WiringError.
    """

    kind: Literal[ArtifactKind.JSON] = ArtifactKind.JSON
    payload: JsonValue


Artifact = Annotated[
    TextArtifact
    | CodeArtifact
    | CodeCandidateSetArtifact
    | IdentifiedCandidateSetArtifact
    | JsonArtifact,
    Field(discriminator="kind"),
]


def parsed_module(code: CodeArtifact) -> ast.Module:
    """Derived view: recomputed on demand. Callers cache (metrics
    ViewCache); never stored on the artifact, so serialization stays
    lossless-by-construction.
    """
    return ast.parse(code.source)
