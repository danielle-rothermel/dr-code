"""Base contract shared by metric operators."""

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
from dr_code.base import FrozenModel
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CodeArtifact,
    TextArtifact,
)

SettingsT = TypeVar("SettingsT", bound=OperatorSettings)


class OperatorResult(FrozenModel):
    """Typed operator output, projected to united facts at the boundary.

    Each result class declares the unit of every field it carries in
    ``UNITS``. Declaring the units next to the fields they describe is what
    lets the record boundary stay mechanical: it never guesses a unit from a
    field name, and a new field without a declared unit fails loudly here
    rather than persisting an unlabelled number.
    """

    UNITS: ClassVar[Mapping[str, MetricFactUnit]] = {}

    def to_facts(self) -> tuple[MetricFact, ...]:
        """Project fields into ordered facts carrying their declared units."""

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
    """Engine services available during phase-two computation."""

    views: ViewCache

    def outcome_for(self, request: ExecutionRequest) -> ExecutionOutcome: ...


class MetricOperator(Generic[SettingsT]):
    """Question implementation managed by the metrics engine."""

    NAME: ClassVar[MetricName]
    # Manual component version. Bump when the operator changes computed
    # facts, execution requests, applicability, defaults, or failure
    # behavior; not for comments, formatting, or behavior-preserving
    # refactors. Stays ``"0"`` while development mode
    # (``[tool.dr-code.component-versioning]`` in ``pyproject.toml``) is
    # enabled.
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
        """Validate domain payloads carried by auxiliary artifacts."""

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
    """Return the canonical text carried by a text-like artifact."""

    if isinstance(value, TextArtifact):
        return value.text
    if isinstance(value, CodeArtifact):
        return value.source
    raise TypeError(f"artifact is not text-like: {value.kind}")
