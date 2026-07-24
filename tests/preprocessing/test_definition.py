"""Tests for PreprocessingDefinition: frozen, hashable, validated."""

from __future__ import annotations

import pytest

from dr_code.eval import (
    PreprocessingDefinition,
    PreprocessingStepBinding,
)
from dr_code.preprocessing.names import StepName


def _norm_definition(
    steps: tuple[PreprocessingStepBinding, ...] = (
        PreprocessingStepBinding(
            instance_name="n", step=StepName.NORMALIZE_LINE_ENDINGS
        ),
    ),
) -> PreprocessingDefinition:
    return PreprocessingDefinition(
        definition_id="d1", version="1", steps=steps
    )


def test_definition_is_frozen() -> None:
    definition = _norm_definition()
    with pytest.raises(Exception):
        definition.definition_id = "other"  # type: ignore[misc]


def test_definition_is_hashable() -> None:
    definition = _norm_definition()
    assert isinstance(hash(definition), int)


def test_definition_hash_is_stable() -> None:
    d = _norm_definition()
    # same definition -> same hash
    assert d.identity_hash() == _norm_definition().identity_hash()


def test_definition_hash_differs_across_versions() -> None:
    a = PreprocessingDefinition(
        definition_id="d1",
        version="1",
        steps=(
            PreprocessingStepBinding(
                instance_name="n", step=StepName.NORMALIZE_LINE_ENDINGS
            ),
        ),
    )
    b = PreprocessingDefinition(
        definition_id="d1",
        version="2",
        steps=(
            PreprocessingStepBinding(
                instance_name="n", step=StepName.NORMALIZE_LINE_ENDINGS
            ),
        ),
    )
    assert a.identity_hash() != b.identity_hash()


def test_definition_hash_differs_across_settings() -> None:
    a = PreprocessingDefinition(
        definition_id="d1",
        version="1",
        steps=(
            PreprocessingStepBinding(
                instance_name="tabs",
                step=StepName.EXPAND_TABS,
                settings={"tab_width": 4},
            ),
        ),
    )
    b = PreprocessingDefinition(
        definition_id="d1",
        version="1",
        steps=(
            PreprocessingStepBinding(
                instance_name="tabs",
                step=StepName.EXPAND_TABS,
                settings={"tab_width": 8},
            ),
        ),
    )
    assert a.identity_hash() != b.identity_hash()


def test_step_spec_settings_default_empty() -> None:
    spec = PreprocessingStepBinding(
        instance_name="n", step=StepName.NORMALIZE_UNICODE
    )
    assert spec.settings == ()


def test_definition_rejects_unknown_step_name() -> None:
    with pytest.raises(ValueError, match="unknown preprocessing step"):
        PreprocessingDefinition(
            definition_id="d",
            version="1",
            steps=(
                PreprocessingStepBinding(
                    instance_name="n", step="not_a_real_step"
                ),
            ),
        )


def test_definition_rejects_reserved_instance_name() -> None:
    with pytest.raises(ValueError, match="reserved"):
        PreprocessingDefinition(
            definition_id="d",
            version="1",
            steps=(
                PreprocessingStepBinding(
                    instance_name="output",
                    step=StepName.NORMALIZE_LINE_ENDINGS,
                ),
            ),
        )


def test_definition_rejects_output_reserved_name() -> None:
    with pytest.raises(ValueError, match="reserved"):
        PreprocessingDefinition(
            definition_id="d",
            version="1",
            steps=(
                PreprocessingStepBinding(
                    instance_name="input",
                    step=StepName.NORMALIZE_LINE_ENDINGS,
                ),
            ),
        )


def test_definition_rejects_duplicate_instance_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        PreprocessingDefinition(
            definition_id="d",
            version="1",
            steps=(
                PreprocessingStepBinding(
                    instance_name="n", step=StepName.NORMALIZE_LINE_ENDINGS
                ),
                PreprocessingStepBinding(
                    instance_name="n", step=StepName.NORMALIZE_UNICODE
                ),
            ),
        )


def test_definition_accepts_unique_non_reserved_names() -> None:
    definition = PreprocessingDefinition(
        definition_id="d",
        version="1",
        steps=(
            PreprocessingStepBinding(
                instance_name="n1", step=StepName.NORMALIZE_LINE_ENDINGS
            ),
            PreprocessingStepBinding(
                instance_name="n2", step=StepName.NORMALIZE_UNICODE
            ),
        ),
    )
    assert len(definition.steps) == 2


def test_definition_serializable_round_trip() -> None:
    definition = PreprocessingDefinition(
        definition_id="d",
        version="2",
        steps=(
            PreprocessingStepBinding(
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
    assert dict(restored.steps[0].settings) == {"tab_width": 2}
