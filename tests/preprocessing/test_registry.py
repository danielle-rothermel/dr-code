"""Tests for the preprocessing step registry."""

from __future__ import annotations

import pytest

from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.registry import REGISTRY
from dr_code.preprocessing.steps.base import Step
from dr_code.preprocessing.steps.extract_candidates import (
    ExtractionStrategy,
    STRATEGY_REGISTRY,
)
from dr_code.trace import ArtifactKind


def test_registry_covers_all_named_steps() -> None:
    assert set(REGISTRY) == {name.value for name in StepName}


def test_registry_is_immutable_after_builtin_registration() -> None:
    with pytest.raises(TypeError):
        REGISTRY["replacement"] = next(iter(REGISTRY.values()))  # type: ignore[index]


def test_registry_values_are_step_subclasses() -> None:
    for step_cls in REGISTRY.values():
        assert issubclass(step_cls, Step)


def test_every_registered_step_declares_classvars() -> None:
    for step_cls in REGISTRY.values():
        assert isinstance(step_cls.NAME, StepName)
        assert isinstance(step_cls.VERSION, str)
        assert isinstance(step_cls.INPUT, ArtifactKind)
        assert isinstance(step_cls.OUTPUT, ArtifactKind)


def test_every_registered_step_has_settings_model() -> None:
    from dr_code.preprocessing.steps.base import StepSettings

    for step_cls in REGISTRY.values():
        assert issubclass(step_cls.Settings, StepSettings)


def test_strategy_registry_covers_all_strategies() -> None:
    assert set(STRATEGY_REGISTRY) == {
        strategy.value for strategy in ExtractionStrategy
    }


def test_strategy_registry_is_immutable_after_builtin_registration() -> None:
    with pytest.raises(TypeError):
        STRATEGY_REGISTRY["replacement"] = next(  # type: ignore[index]
            iter(STRATEGY_REGISTRY.values())
        )
