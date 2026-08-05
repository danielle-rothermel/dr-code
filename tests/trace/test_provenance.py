from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from dr_code.core.models import FrozenModel
from dr_code.trace import (
    ComponentCoordinate,
    ComponentSetting,
    EXTERNAL_PRODUCER,
    ExternalPreprocessingTraceProducer,
    ExternalTraceProducer,
    PreprocessingDefinitionCoordinate,
    PreprocessingTraceProducer,
    StepCoordinate,
    TraceProducer,
    coordinate_settings,
)


def test_external_producer_has_canonical_coordinate() -> None:
    assert EXTERNAL_PRODUCER == ExternalTraceProducer()
    assert EXTERNAL_PRODUCER.model_dump(mode="json") == {"kind": "external"}


def test_trace_producer_carries_declared_manual_version() -> None:
    producer = PreprocessingTraceProducer(
        definition=PreprocessingDefinitionCoordinate(
            definition_id="preprocessing-definition",
            version="test-version",
            steps=(
                StepCoordinate(
                    instance_name="normalize",
                    component=ComponentCoordinate(
                        registered_name="expand_tabs",
                        version="step-version",
                        settings=(
                            ComponentSetting(name="tab_width", value=2),
                        ),
                    ),
                ),
            ),
        )
    )

    assert producer.model_dump(mode="json") == {
        "kind": "preprocessing",
        "definition": {
            "definition_id": "preprocessing-definition",
            "version": "test-version",
            "steps": [
                {
                    "instance_name": "normalize",
                    "component": {
                        "registered_name": "expand_tabs",
                        "version": "step-version",
                        "settings": [{"name": "tab_width", "value": 2}],
                    },
                }
            ],
        },
    }


def test_external_preprocessing_producer_round_trips_through_type_adapter() -> (
    None
):
    producer = ExternalPreprocessingTraceProducer(
        definition=PreprocessingDefinitionCoordinate(
            definition_id="external-preprocessing-definition",
            version="external-version",
            steps=(),
        )
    )
    adapter = TypeAdapter(TraceProducer)

    restored = adapter.validate_json(adapter.dump_json(producer))

    assert restored == producer
    assert isinstance(restored, ExternalPreprocessingTraceProducer)


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "external", "version": "0"},
        {"kind": "preprocessing"},
        {"kind": "preprocessing", "definition": None},
    ],
)
def test_trace_producer_rejects_incomplete_or_mixed_variants(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(TraceProducer).validate_python(payload)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_component_setting_rejects_non_finite_float(value: float) -> None:
    with pytest.raises(ValidationError):
        ComponentSetting(name="threshold", value=value)


def test_component_setting_finite_float_round_trips_through_json() -> None:
    setting = ComponentSetting(name="threshold", value=0.5)

    assert ComponentSetting.model_validate_json(setting.model_dump_json()) == (
        setting
    )


def test_coordinate_settings_rejects_non_string_tuple() -> None:
    class _IntTupleSettings(FrozenModel):
        alternatives: tuple[int, ...] = (1, 2)

    with pytest.raises(
        TypeError,
        match="unsupported persisted tuple setting for 'alternatives'",
    ):
        coordinate_settings(_IntTupleSettings())


def test_coordinate_settings_rejects_unsupported_value_type() -> None:
    class _MappingSettings(FrozenModel):
        mapping: dict[str, str] = {}

    with pytest.raises(
        TypeError,
        match="unsupported persisted setting shape for 'mapping': dict",
    ):
        coordinate_settings(_MappingSettings())
