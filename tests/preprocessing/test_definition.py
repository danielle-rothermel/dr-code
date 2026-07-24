"""Tests for frozen, validated preprocessing definitions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dr_code.preprocessing.definition import (
    PreprocessingDefinition,
    StepSpec,
)
from dr_code.preprocessing.names import StepName
from dr_code.preprocessing.registry import REGISTRY
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


# ---------------------------------------------------------------------------
# Settings belong to the named step; the discriminator is required.
# ---------------------------------------------------------------------------


def test_step_spec_rejects_settings_from_another_step() -> None:
    """Another step's settings model is revalidated, not waved through."""
    with pytest.raises(ValidationError):
        StepSpec(
            instance_name="u",
            step=StepName.NORMALIZE_UNICODE,
            settings=ExpandTabsSettings(tab_width=8),
        )


def test_step_spec_accepts_the_named_steps_settings_instance() -> None:
    spec = StepSpec(
        instance_name="e",
        step=StepName.EXPAND_TABS,
        settings=ExpandTabsSettings(tab_width=8),
    )
    assert spec.settings == ExpandTabsSettings(tab_width=8)


def test_step_spec_accepts_plain_dict_settings() -> None:
    spec = StepSpec(
        instance_name="e",
        step=StepName.EXPAND_TABS,
        settings={"tab_width": 3},
    )
    assert spec.settings == ExpandTabsSettings(tab_width=3)


def test_step_spec_missing_step_raises_validation_error() -> None:
    """A payload without the discriminator gets pydantic's missing-field
    error, never a bare KeyError past the validation boundary."""
    with pytest.raises(ValidationError):
        StepSpec.model_validate({"instance_name": "n", "settings": {}})


# ---------------------------------------------------------------------------
# The validator reaches every registered step, not just the settings-bearing
# ones. A step with no settings must reject settings rather than ignore them.
# ---------------------------------------------------------------------------


#: Every registered step, so a newly registered step is covered here the
#: moment it exists rather than whenever someone remembers to add a case.
_ALL_STEPS = sorted(REGISTRY, key=str)


@pytest.mark.parametrize("step_name", _ALL_STEPS)
def test_every_registered_step_resolves_its_settings_model(
    step_name: str,
) -> None:
    spec = StepSpec.model_validate(
        {"instance_name": "s", "step": step_name, "settings": {}}
    )
    assert isinstance(spec.settings, REGISTRY[step_name].Settings)


@pytest.mark.parametrize("step_name", _ALL_STEPS)
def test_every_registered_step_rejects_an_unknown_setting(
    step_name: str,
) -> None:
    # Persisted setting names are a wire contract, so an unrecognized name
    # is an error at the boundary rather than a silently dropped field.
    with pytest.raises(ValidationError):
        StepSpec.model_validate(
            {
                "instance_name": "s",
                "step": step_name,
                "settings": {"not_a_real_setting": 1},
            }
        )


@pytest.mark.parametrize(
    "step_name",
    [
        StepName.EXTRACT_CANDIDATES,
        StepName.EXPAND_LAST_RETURN_SALVAGE,
        StepName.IDENTIFY_CANDIDATES,
        StepName.FILTER_COMPILABLE,
        StepName.FILTER_PLAIN_LITERAL,
        StepName.FILTER_CODE_REPR,
        StepName.FILTER_HAS_TOP_LEVEL_FUNCTION,
        StepName.FILTER_NONBLANK_CANDIDATES,
        StepName.MATERIALIZE_CANDIDATES,
        StepName.REQUIRE_NONBLANK_TEXT,
        StepName.RETURN_ALL,
    ],
)
def test_exhaustive_pipeline_steps_take_no_settings(
    step_name: StepName,
) -> None:
    # These steps are deliberately unconfigurable: extraction is exhaustive
    # and each filter applies one fixed predicate, so their persisted
    # coordinates carry an empty settings projection.
    spec = StepSpec(instance_name="s", step=step_name)
    assert spec.settings.model_dump() == {}


def test_step_spec_rejects_expand_tabs_settings_on_a_filter_step() -> None:
    # Wrong-component settings are rejected even when the target step has
    # no settings of its own to collide with.
    with pytest.raises(ValidationError):
        StepSpec(
            instance_name="f",
            step=StepName.FILTER_COMPILABLE,
            settings=ExpandTabsSettings(tab_width=8),
        )
