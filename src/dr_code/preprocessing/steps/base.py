"""Preprocessing step base classes.

Mirrors ``synthetic/corruptions/base.py``: a uniform base class with a
``NAME`` classvar and an ``apply`` method. One difference — preprocessing
steps take **no rng**: determinism is settings + input only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Generic, TypeVar, cast

from pydantic import JsonValue

from dr_code.models import FrozenModel
from dr_code.preprocessing.failures import PreprocessingFailureCode
from dr_code.preprocessing.names import StepName
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CodeCandidateSetArtifact,
)


class StepSettings(FrozenModel):
    """Base for per-step settings; each ``Step`` subclass declares its own.

    A step with no tunables uses this empty base directly.
    """


SettingsT = TypeVar("SettingsT", bound=StepSettings)


class StepFailedError(Exception):
    """A data failure: the step cannot produce output without guessing.

    Converted to ``Absent``-with-cause by the runner; never escapes
    ``run_preprocessing``. Distinct from ``WiringError`` (a definition bug
    raised at bind time).
    """

    def __init__(
        self,
        cause: str,
        *,
        failure_code: PreprocessingFailureCode,
        facts: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(cause)
        if not isinstance(failure_code, PreprocessingFailureCode):
            raise TypeError("failure_code must be a PreprocessingFailureCode")
        self.cause: str = cause
        self.failure_code = failure_code
        self.facts: dict[str, JsonValue] = (
            dict(facts) if facts is not None else {}
        )


@dataclass(frozen=True, slots=True)
class StepOutput:
    """One step's deterministic result.

    ``facts`` describe the output (chosen alternative, rejection reasons,
    candidate counts); a step may describe, never prefer.
    """

    value: Artifact
    facts: Mapping[str, JsonValue] = field(default_factory=dict)


class Step(Generic[SettingsT]):
    """Base class for preprocessing steps.

    Subclasses declare ``NAME``, ``VERSION``, ``INPUT``/``OUTPUT`` kinds,
    and a ``Settings`` model. ``apply`` is deterministic: same settings +
    input => same output (no RNG, ambient state, or clocks).
    """

    NAME: ClassVar[StepName]
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
    """

    INPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET
    OUTPUT: ClassVar[ArtifactKind] = ArtifactKind.CODE_CANDIDATE_SET

    def apply_to_candidate(self, source: str) -> str | list[str]:
        raise NotImplementedError

    def apply(self, value: Artifact) -> StepOutput:
        candidates = _candidate_set(value).candidates
        candidate_set = _candidate_set(value)
        mapped: list[str] = []
        lineage = []
        for index, candidate in enumerate(candidates):
            result = self.apply_to_candidate(candidate)
            if isinstance(result, list):
                mapped.extend(result)
                if candidate_set.lineage:
                    lineage.extend(
                        candidate_set.lineage_at(index).model_copy(
                            update={"candidate_id": None}
                        )
                        for _ in result
                    )
            else:
                mapped.append(result)
                if candidate_set.lineage:
                    lineage.append(
                        candidate_set.lineage_at(index).model_copy(
                            update={"candidate_id": None}
                        )
                    )
        return StepOutput(
            value=CodeCandidateSetArtifact(
                candidates=tuple(mapped), lineage=tuple(lineage)
            ),
            facts={
                "input_candidate_count": len(candidates),
                "output_candidate_count": len(mapped),
            },
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
            "no alternative produced candidates",
            failure_code=PreprocessingFailureCode.NO_ALTERNATIVE_CANDIDATES,
        )


def _candidate_set(value: Artifact) -> CodeCandidateSetArtifact:
    """Narrow an Artifact to its candidate-set member."""
    assert isinstance(value, CodeCandidateSetArtifact)
    return value


__all__ = [
    "AlternativesStep",
    "CandidateMapStep",
    "Step",
    "StepFailedError",
    "StepOutput",
    "StepSettings",
]
