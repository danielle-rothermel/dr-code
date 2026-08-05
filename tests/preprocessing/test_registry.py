"""Tests for the preprocessing step registry."""

from __future__ import annotations

from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.registry import REGISTRY
from dr_code.preprocessing.steps.base import Step
from dr_code.preprocessing.steps.extract_all_representations import (
    _READINGS,
    Representation,
)
from dr_code.trace import ArtifactKind


def test_registry_covers_all_named_steps() -> None:
    assert set(REGISTRY) == {name.value for name in StepName}


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


def test_every_representation_has_exactly_one_reading() -> None:
    # Every declared representation is read, each exactly once: a member
    # with no reading would be a name for something never extracted.
    read = [representation for representation, _reading in _READINGS]
    assert read == list(Representation)
