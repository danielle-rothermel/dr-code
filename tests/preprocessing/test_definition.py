"""Tests for frozen, validated preprocessing definitions."""

from __future__ import annotations

import pytest

from dr_code.preprocessing.definition import (
    PreprocessingDefinition,
    StepSpec,
)
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.steps.base import StepSettings
from dr_code.preprocessing.steps.expand_tabs import ExpandTabsSettings
from dr_code.trace import WiringError


def _norm_definition(
    steps: tuple[StepSpec, ...] = (
        StepSpec(instance_name="n", step=StepName.NORMALIZE_LINE_ENDINGS),
    ),
) -> PreprocessingDefinition:
    return PreprocessingDefinition(
        definition_id="d1", version="1", steps=steps
    )


def test_definition_is_frozen() -> None:
    definition = _norm_definition()
    with pytest.raises(Exception):
        definition.definition_id = "other"  # type: ignore[misc]


def test_definition_is_not_hashable() -> None:
    with pytest.raises(TypeError, match="unhashable type"):
        hash(_norm_definition())


def test_step_spec_settings_default_empty() -> None:
    spec = StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE)
    assert spec.settings == StepSettings()


def test_step_spec_rejects_unknown_step_name() -> None:
    with pytest.raises(Exception):
        StepSpec(instance_name="n", step="not_a_real_step")  # type: ignore[arg-type]


def test_definition_rejects_reserved_instance_name() -> None:
    with pytest.raises(WiringError):
        PreprocessingDefinition(
            definition_id="d",
            version="1",
            steps=(
                StepSpec(
                    instance_name="output",
                    step=StepName.NORMALIZE_LINE_ENDINGS,
                ),
            ),
        )


def test_definition_rejects_output_reserved_name() -> None:
    with pytest.raises(WiringError):
        PreprocessingDefinition(
            definition_id="d",
            version="1",
            steps=(
                StepSpec(
                    instance_name="input",
                    step=StepName.NORMALIZE_LINE_ENDINGS,
                ),
            ),
        )


def test_definition_rejects_duplicate_instance_names() -> None:
    with pytest.raises(WiringError):
        PreprocessingDefinition(
            definition_id="d",
            version="1",
            steps=(
                StepSpec(
                    instance_name="n", step=StepName.NORMALIZE_LINE_ENDINGS
                ),
                StepSpec(instance_name="n", step=StepName.NORMALIZE_UNICODE),
            ),
        )


def test_definition_accepts_unique_non_reserved_names() -> None:
    definition = PreprocessingDefinition(
        definition_id="d",
        version="1",
        steps=(
            StepSpec(instance_name="n1", step=StepName.NORMALIZE_LINE_ENDINGS),
            StepSpec(instance_name="n2", step=StepName.NORMALIZE_UNICODE),
        ),
    )
    assert len(definition.steps) == 2


def test_definition_serializable_round_trip() -> None:
    definition = PreprocessingDefinition(
        definition_id="d",
        version="2",
        steps=(
            StepSpec(
                instance_name="e",
                step=StepName.EXPAND_TABS,
                settings={"tab_width": 2},
            ),
        ),
    )
    restored = PreprocessingDefinition.model_validate_json(
        definition.model_dump_json()
    )
    assert restored == definition
    assert restored.steps[0].settings == ExpandTabsSettings(tab_width=2)
