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


class CandidateOrigin(FrozenModel):
    """One decoder-text variant and discovery rule that yielded a candidate."""

    variant: str
    strategy: str


class CandidateLineage(FrozenModel):
    """Stable post-cleaning identity plus every extraction origin."""

    candidate_id: str | None = None
    origins: tuple[CandidateOrigin, ...] = ()


class CodeCandidateSetArtifact(FrozenModel):
    """Ordered candidate sources, conservative first. Fan-out as data
    (P-S2)."""

    kind: Literal[ArtifactKind.CODE_CANDIDATE_SET] = (
        ArtifactKind.CODE_CANDIDATE_SET
    )
    candidates: tuple[str, ...]
    lineage: tuple[CandidateLineage, ...] = ()

    @model_validator(mode="after")
    def _validate_lineage_alignment(self) -> CodeCandidateSetArtifact:
        if self.lineage and len(self.lineage) != len(self.candidates):
            raise ValueError(
                "candidate lineage must be empty or aligned with candidates"
            )
        return self

    def lineage_at(self, index: int) -> CandidateLineage:
        """Return aligned lineage, or an empty record for legacy producers."""
        if not self.lineage:
            return CandidateLineage()
        return self.lineage[index]


class JsonArtifact(FrozenModel):
    """Escape hatch for externally built traces (X-S2), e.g. a HumanEval
    task payload for code_test. Consumers revalidate `payload` into their
    own model at bind time; a payload that fails validation is a
    WiringError.
    """

    kind: Literal[ArtifactKind.JSON] = ArtifactKind.JSON
    payload: JsonValue


Artifact = Annotated[
    TextArtifact | CodeArtifact | CodeCandidateSetArtifact | JsonArtifact,
    Field(discriminator="kind"),
]


def parsed_module(code: CodeArtifact) -> ast.Module:
    """Derived view: recomputed on demand. Callers cache (metrics
    ViewCache); never stored on the artifact, so serialization stays
    lossless-by-construction.
    """
    return ast.parse(code.source)
