from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, Generic, Protocol, TypeVar, runtime_checkable

from dr_code.evaluation.candidate_job import CandidateEvaluatorSuite
from dr_code.metrics.coordinates import MetricQuestionCoordinate
from dr_code.metrics.engine.views import ViewCache
from dr_code.metrics.names import MetricName
from dr_code.metrics.records import MetricValue
from dr_code.metrics.settings import OperatorSettings
from dr_code.metrics.units import MetricValueUnit
from dr_code.core.models import FrozenModel
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CodeArtifact,
    TextArtifact,
)

SettingsT = TypeVar("SettingsT", bound=OperatorSettings)


class OperatorResult(FrozenModel):
    UNITS: ClassVar[Mapping[str, MetricValueUnit]] = {}

    def to_values(self) -> tuple[MetricValue, ...]:
        values: list[MetricValue] = []
        for name, value in self.model_dump(mode="python").items():
            unit = type(self).UNITS.get(name)
            if unit is None:
                raise ValueError(
                    f"{type(self).__name__} declares no unit for value {name!r}"
                )
            values.append(MetricValue(name=name, value=value, unit=unit))
        return tuple(values)


class EngineContext(Protocol):
    views: ViewCache
    question: MetricQuestionCoordinate
    candidate_execution_outcome: object | None


@runtime_checkable
class CandidateExecutableMetric(Protocol):
    INJECTS_CANDIDATE_SOURCE: ClassVar[bool]

    def evaluator_suites(
        self,
        value: Artifact,
        aux: Mapping[str, Artifact],
        question: MetricQuestionCoordinate,
        /,
    ) -> tuple[CandidateEvaluatorSuite, ...]: ...


class MetricOperator(Generic[SettingsT]):
    NAME: ClassVar[MetricName]
    # In development mode, keep VERSION at "0". Afterward, bump it for value,
    # request, applicability, default, or failure changes.
    VERSION: ClassVar[str]
    INPUT: ClassVar[ArtifactKind]
    ACCEPTED_INPUTS: ClassVar[frozenset[ArtifactKind]]
    INJECTS_CANDIDATE_SOURCE: ClassVar[bool] = False
    Settings: ClassVar[type[OperatorSettings]] = OperatorSettings

    def __init__(self, settings: SettingsT) -> None:
        self.settings: SettingsT = settings

    @classmethod
    def accepted_input_kinds(cls) -> frozenset[ArtifactKind]:
        return getattr(cls, "ACCEPTED_INPUTS", frozenset({cls.INPUT}))

    def auxiliary_keys(self) -> tuple[str, ...]:
        return ()

    def accepted_auxiliary_kinds(
        self,
        key: str,
    ) -> frozenset[ArtifactKind]:
        _ = key
        return frozenset(ArtifactKind)

    def validate_auxiliary(self, aux: Mapping[str, Artifact]) -> None:
        _ = aux

    def compute(
        self,
        value: Artifact,
        aux: Mapping[str, Artifact],
        ctx: EngineContext,
    ) -> OperatorResult:
        raise NotImplementedError


def artifact_text(value: Artifact) -> str:
    if isinstance(value, TextArtifact):
        return value.text
    if isinstance(value, CodeArtifact):
        return value.source
    raise TypeError(f"artifact is not text-like: {value.kind}")
