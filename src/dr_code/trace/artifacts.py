"""Artifact discriminated union + derived-view helpers."""

from __future__ import annotations

import ast
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, model_validator

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
    """Python source text.

    The type itself carries no compilation guarantee; producers and consumers
    validate source at the boundaries that require it. The source string is
    canonical, and the AST is a derived view.
    """

    kind: Literal[ArtifactKind.CODE] = ArtifactKind.CODE
    source: str


class ExtractionOperation(FrozenModel):
    """One named operation that produced or transformed a candidate.

    ``operation_name`` is a producer-owned vocabulary: the trace layer
    records the name a producer stamped and never interprets it.
    """

    operation_name: str


class CandidateOrigin(FrozenModel):
    """One origin of a candidate: an operation plus where it read from.

    ``input_location`` names the position in that operation's input the
    candidate came from — for a block-extracting operation, the ordinal of
    the block within the text; for an elementwise transform, the ordinal of
    the candidate it was applied to. Lineage is extended by appending
    origins, never by replacing them.
    """

    operation: ExtractionOperation
    input_location: int = Field(ge=0)


class CodeCandidate(FrozenModel):
    """One candidate source with its complete ordered lineage.

    ``origins`` is non-empty and ordered oldest-first: the operation that
    first produced the source, then every operation that transformed it.
    Candidate identity within a trace is exact-source equality plus position
    in the containing set — the record carries no hash or semantic id.
    """

    source: str
    origins: tuple[CandidateOrigin, ...] = Field(min_length=1)

    def extended(
        self, origin: CandidateOrigin, *, source: str
    ) -> CodeCandidate:
        """A candidate with ``source`` and ``origin`` appended to lineage."""
        return CodeCandidate(source=source, origins=(*self.origins, origin))


class CodeCandidateSetArtifact(FrozenModel):
    """Ordered candidate records, with conservative candidates first."""

    kind: Literal[ArtifactKind.CODE_CANDIDATE_SET] = (
        ArtifactKind.CODE_CANDIDATE_SET
    )
    candidates: tuple[CodeCandidate, ...]


class CandidateInspection(FrozenModel):
    """Structural facts about one candidate source.

    Structure only: whether the source parses, whether it compiles, the
    error text when it does not, and the names of its top-level functions.
    Whether those facts make a candidate acceptable is a policy verdict,
    owned by filter steps rather than by the trace layer.

    The three structural invariants below hold for every inspection this
    package produces and are enforced on load, so a trace supplied from
    outside cannot carry an inspection that describes an impossible
    source: source that does not parse cannot compile, and each error
    text is present exactly when its own outcome failed.
    """

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
    """One candidate paired with the structural inspection of its source."""

    candidate: CodeCandidate
    inspection: CandidateInspection


class InspectedCodeCandidateSetArtifact(FrozenModel):
    """Ordered inspected candidates, in the containing set's order."""

    kind: Literal[ArtifactKind.INSPECTED_CODE_CANDIDATE_SET] = (
        ArtifactKind.INSPECTED_CODE_CANDIDATE_SET
    )
    candidates: tuple[InspectedCodeCandidate, ...]


class JsonArtifact(FrozenModel):
    """JSON payload for externally built traces, such as a HumanEval task.

    Consumers revalidate ``payload`` into their own model at bind time; a
    payload that fails validation is a ``WiringError``.
    """

    kind: Literal[ArtifactKind.JSON] = ArtifactKind.JSON
    payload: JsonValue


Artifact = Annotated[
    TextArtifact
    | CodeArtifact
    | CodeCandidateSetArtifact
    | InspectedCodeCandidateSetArtifact
    | JsonArtifact,
    Field(discriminator="kind"),
]


def parsed_module(code: CodeArtifact) -> ast.Module:
    """Derived view: recomputed on demand. Callers cache (metrics
    ViewCache); never stored on the artifact, so serialization stays
    lossless-by-construction.
    """
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
