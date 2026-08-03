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
    """Python source text.

    The type itself carries no compilation guarantee; producers and consumers
    validate source at the boundaries that require it. The source string is
    canonical, and the AST is a derived view.
    """

    kind: Literal[ArtifactKind.CODE] = ArtifactKind.CODE
    source: str


class CodeCandidateSetArtifact(FrozenModel):
    """Ordered candidate sources, with conservative candidates first."""

    kind: Literal[ArtifactKind.CODE_CANDIDATE_SET] = (
        ArtifactKind.CODE_CANDIDATE_SET
    )
    candidates: tuple[str, ...]


class JsonArtifact(FrozenModel):
    """JSON payload for externally built traces, such as a HumanEval task.

    Consumers revalidate ``payload`` into their own model at bind time; a
    payload that fails validation is a ``WiringError``.
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
