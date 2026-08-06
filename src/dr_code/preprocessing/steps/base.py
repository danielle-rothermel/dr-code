from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Generic, TypeVar, cast

from dr_code.core.models import FrozenModel
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CandidateOrigin,
    CodeCandidate,
    CodeCandidateSetArtifact,
    ExtractionOperation,
    JsonFactValue,
)
from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName


class StepSettings(FrozenModel):
    pass


SettingsT = TypeVar("SettingsT", bound=StepSettings)


class StepFailedError(Exception):
    def __init__(
        self,
        code: PreprocessingFailureCode,
        cause: str,
        *,
        evidence: Mapping[str, JsonFactValue] | None = None,
    ) -> None:
        super().__init__(cause)
        self.code = code
        self.cause = cause
        self.evidence: Mapping[str, JsonFactValue] = evidence or {}


@dataclass(frozen=True, slots=True)
class StepOutput:
    value: Artifact
    facts: Mapping[str, JsonFactValue] = field(default_factory=dict)


class Step(Generic[SettingsT]):
    NAME: ClassVar[StepName]
    # In development mode, keep VERSION at "0". Afterward, bump it for
    # accepted-input, output, fact, default, or failure changes, including
    # runtime or dependency changes that can alter those behaviors.
    VERSION: ClassVar[str]
    INPUT: ClassVar[ArtifactKind]
    OUTPUT: ClassVar[ArtifactKind]
    Settings: ClassVar[type[StepSettings]] = StepSettings

    def __init__(self, settings: SettingsT | None = None) -> None:
        self.settings: SettingsT = (
            settings
            if settings is not None
            else cast(SettingsT, self.Settings())
        )

    def apply(self, value: Artifact) -> StepOutput:  # pragma: no cover
        raise NotImplementedError


def candidate_set(value: Artifact) -> CodeCandidateSetArtifact:
    assert isinstance(value, CodeCandidateSetArtifact)
    return value


class CandidateMapStep(Step[StepSettings]):
    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply_to_candidate(self, source: str) -> str | list[str]:
        raise NotImplementedError

    def apply(self, value: Artifact) -> StepOutput:
        operation = ExtractionOperation(operation_name=self.NAME.value)
        mapped: list[CodeCandidate] = []
        for index, candidate in enumerate(candidate_set(value).candidates):
            origin = CandidateOrigin(operation=operation, input_location=index)
            result = self.apply_to_candidate(candidate.source)
            sources = result if isinstance(result, list) else [result]
            mapped.extend(
                candidate.extended(origin, source=source) for source in sources
            )
        return StepOutput(
            value=CodeCandidateSetArtifact(candidates=tuple(mapped))
        )


__all__ = [
    "CandidateMapStep",
    "Step",
    "StepFailedError",
    "StepOutput",
    "StepSettings",
    "candidate_set",
]
