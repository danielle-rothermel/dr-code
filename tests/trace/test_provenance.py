"""Producer provenance contracts."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from dr_code.trace import (
    ComponentCoordinate,
    ComponentSetting,
    EXTERNAL_PRODUCER,
    ExternalTraceProducer,
    PreprocessingDefinitionCoordinate,
    PreprocessingTraceProducer,
    StepCoordinate,
    TraceProducer,
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
