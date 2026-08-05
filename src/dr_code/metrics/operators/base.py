"""Base contract shared by metric operators."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Generic, Protocol, TypeVar

from dr_code.metrics.engine.execution import (
    ExecutionOutcome,
    ExecutionRequest,
)
from dr_code.metrics.engine.views import ViewCache
from dr_code.metrics.names import MetricName
from dr_code.base import FrozenModel
from dr_code.trace import (
    Artifact,
    ArtifactKind,
    CodeArtifact,
    TextArtifact,
)

if TYPE_CHECKING:
    # ``records`` imports ``definition``, which imports ``OperatorSettings``
    # from here; the annotation-only use stays out of the runtime cycle.
    from dr_code.metrics.records import MetricScalar


class OperatorSettings(FrozenModel):
    """Validated parameters that determine an operator's semantics."""


SettingsT = TypeVar("SettingsT", bound=OperatorSettings)


class OperatorResult(FrozenModel):
    """Typed operator output; flattened to record values at the record boundary."""

    def to_values(self) -> dict[str, MetricScalar]:
        return self.model_dump(mode="python")


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
