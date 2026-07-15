"""Artifact discriminated union + derived-view helpers."""

from __future__ import annotations

import ast
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, JsonValue

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


class CodeCandidateSetArtifact(FrozenModel):
    """Ordered candidate sources, conservative first. Fan-out as data
    (P-S2)."""

    kind: Literal[ArtifactKind.CODE_CANDIDATE_SET] = (
        ArtifactKind.CODE_CANDIDATE_SET
    )
    candidates: tuple[str, ...]


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
