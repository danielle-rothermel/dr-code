from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, Generic, Protocol, TypeVar

from dr_code.metrics.engine.execution import (
    ExecutionOutcome,
    ExecutionRequest,
)
from dr_code.metrics.engine.views import ViewCache
from dr_code.metrics.names import MetricName
from dr_code.metrics.records import MetricFact
from dr_code.metrics.settings import OperatorSettings
from dr_code.metrics.units import MetricFactUnit
from dr_code.core.models import FrozenModel
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CodeArtifact,
    TextArtifact,
)

SettingsT = TypeVar("SettingsT", bound=OperatorSettings)


class OperatorResult(FrozenModel):
    UNITS: ClassVar[Mapping[str, MetricFactUnit]] = {}

    def to_facts(self) -> tuple[MetricFact, ...]:
        facts: list[MetricFact] = []
        for name, value in self.model_dump(mode="python").items():
            unit = type(self).UNITS.get(name)
            if unit is None:
                raise ValueError(
                    f"{type(self).__name__} declares no unit for fact {name!r}"
                )
            facts.append(MetricFact(name=name, value=value, unit=unit))
        return tuple(facts)


class EngineContext(Protocol):
    views: ViewCache

    def outcome_for(self, request: ExecutionRequest) -> ExecutionOutcome: ...


class MetricOperator(Generic[SettingsT]):
    NAME: ClassVar[MetricName]
    # In development mode, keep VERSION at "0". Afterward, bump it for fact,
    # request, applicability, default, or failure changes.
    VERSION: ClassVar[str]
    INPUT: ClassVar[ArtifactKind]
    ACCEPTED_INPUTS: ClassVar[frozenset[ArtifactKind]]
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

    def execution_requests(
        self,
        value: Artifact,
        aux: Mapping[str, Artifact],
    ) -> tuple[ExecutionRequest, ...]:
        _ = value, aux
        return ()

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
