"""Preprocessing step base classes.

Mirrors ``synthetic/corruptions/base.py``: a uniform base class with a
``NAME`` classvar and an ``apply`` method. One difference — preprocessing
steps take **no rng**: determinism is settings + input only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import UNIQUE, StrEnum, verify
from typing import ClassVar, Generic, TypeVar, cast

from dr_code.base import FrozenModel
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CandidateOrigin,
    CodeCandidate,
    CodeCandidateSetArtifact,
    ExtractionOperation,
    JsonFactValue,
)
from dr_code.preprocessing.names import StepName


@verify(UNIQUE)
class FailureCode(StrEnum):
    """The failure vocabulary preprocessing steps raise.

    Preprocessing owns these codes; the trace layer stores whichever string
    a producer stamps into ``Absent.failure_code`` without interpreting it.

    Never build a payload by iterating this enum: the set of members is a
    closed vocabulary, not an ordered list, and its iteration order is not
    part of any persisted format. Reference members individually by name.
    """

    #: No alternative in a step's first-success ladder produced candidates.
    NO_ALTERNATIVE_PRODUCED_CANDIDATES = "no_alternative_produced_candidates"
    #: The configured field marker is absent from the input.
    MISSING_FIELD_MARKER = "missing_field_marker"
    #: The configured field marker is present but its value is empty.
    EMPTY_FIELD_MARKER_VALUE = "empty_field_marker_value"
    #: Every extracted candidate was dropped by the structural filters.
    NO_CANDIDATE_SURVIVED_FILTERING = "no_candidate_survived_filtering"


class StepSettings(FrozenModel):
    """Base for per-step settings; each ``Step`` subclass declares its own.

    A step with no tunables uses this empty base directly.
    """


SettingsT = TypeVar("SettingsT", bound=StepSettings)


class StepFailedError(Exception):
    """A data failure: the step cannot produce output without guessing.

    Raised with a ``FailureCode`` member naming the failure kind plus a
    free-form ``cause`` detailing it; ``code`` is stored as that member's
    plain string, the form the trace layer records. Converted to
    ``Absent`` by the runner, which copies both through; never escapes
    ``run_preprocessing``.
    Distinct from ``WiringError`` (a definition bug raised at bind time).
    """

    def __init__(self, code: FailureCode, cause: str) -> None:
        super().__init__(cause)
        self.code = code.value
        self.cause = cause


@dataclass(frozen=True, slots=True)
class StepOutput:
    """One step's deterministic result.

    ``facts`` describe the output (chosen alternative, rejection reasons,
    candidate counts) as finite JSON values; a step may describe, never
    prefer.
    """

    value: Artifact
    facts: Mapping[str, JsonFactValue] = field(default_factory=dict)


class Step(Generic[SettingsT]):
    """Base class for preprocessing steps.

    Subclasses declare ``NAME``, ``VERSION``, ``INPUT``/``OUTPUT`` kinds,
    and a ``Settings`` model. ``apply`` is deterministic: same settings +
    input => same output (no RNG, ambient state, or clocks).
    """

    NAME: ClassVar[StepName]
    # Manual component version. Bump when the step changes accepted inputs,
    # produced outputs or facts, defaults, or failure behavior; not for
    # comments, formatting, or behavior-preserving refactors. Stays ``"0"``
    # while development mode (``[tool.dr-code.component-versioning]`` in
    # ``pyproject.toml``) is enabled.
    VERSION: ClassVar[str]
    INPUT: ClassVar[ArtifactKind]
    OUTPUT: ClassVar[ArtifactKind]
    Settings: ClassVar[type[StepSettings]] = StepSettings

    def __init__(self, settings: SettingsT | None = None) -> None:
        # Optional so steps with no tunables instantiate as ``StepCls()``;
        # bind_definition always passes explicit validated settings.
        self.settings: SettingsT = (
            settings
            if settings is not None
            else cast(SettingsT, self.Settings())
        )

    def apply(self, value: Artifact) -> StepOutput:  # pragma: no cover
        raise NotImplementedError


class CandidateMapStep(Step[StepSettings]):
    """Elementwise ``CandidateSet -> CandidateSet``; fan-out stays data.

    ``apply`` maps over candidates in order, flattening list results in
    place so splits (e.g. name-guard) preserve relative candidate order.
    Each produced candidate's lineage is extended with this step's
    operation and the ordinal of the input candidate it came from, so a
    split's parts all record the candidate they were split out of.
    """

    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply_to_candidate(self, source: str) -> str | list[str]:
        raise NotImplementedError

    def apply(self, value: Artifact) -> StepOutput:
        operation = ExtractionOperation(operation_name=self.NAME.value)
        mapped: list[CodeCandidate] = []
        for index, candidate in enumerate(_candidate_set(value).candidates):
            origin = CandidateOrigin(operation=operation, input_location=index)
            result = self.apply_to_candidate(candidate.source)
            sources = result if isinstance(result, list) else [result]
            mapped.extend(
                candidate.extended(origin, source=source) for source in sources
            )
        return StepOutput(
            value=CodeCandidateSetArtifact(candidates=tuple(mapped))
        )


class AlternativesStep(Step[SettingsT], Generic[SettingsT]):
    """Ordered first-success ladder inside one step — never pipeline branches.

    ``apply`` tries alternatives conservative-first; the winner's name is
    recorded as ``facts["alternative"]``; all-fail raises
    ``StepFailedError``.
    """

    def alternatives(
        self,
    ) -> tuple[tuple[str, Callable[[Artifact], Artifact | None]], ...]:
        raise NotImplementedError

    def apply(self, value: Artifact) -> StepOutput:
        for name, strategy_fn in self.alternatives():
            result = strategy_fn(value)
            if result is not None:
                return StepOutput(value=result, facts={"alternative": name})
        raise StepFailedError(
            FailureCode.NO_ALTERNATIVE_PRODUCED_CANDIDATES,
            "no alternative produced candidates",
        )


def _candidate_set(value: Artifact) -> CodeCandidateSetArtifact:
    """Narrow an Artifact to its candidate-set member."""
    assert isinstance(value, CodeCandidateSetArtifact)
    return value


__all__ = [
    "AlternativesStep",
    "CandidateMapStep",
    "FailureCode",
    "Step",
    "StepFailedError",
    "StepOutput",
    "StepSettings",
]
