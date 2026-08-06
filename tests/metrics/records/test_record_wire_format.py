from __future__ import annotations

import json

import pytest

from ._builders import (
    _fact,
    _identity,
    _measured,
    _not_applicable,
    _operator_failure,
    _question_coordinate,
)


# Literal keys pin persisted record shapes; deriving them would hide drift.
_GOLDEN_MEASURED_RECORD = {
    "schema_version": 1,
    "status": "measured",
    "identity": {
        "question": {
            "metric": "code_leakage",
            "on_key": "output",
            "settings": [
                {"name": "task_names", "value": ["add_one"]},
            ],
        },
        "metric_version": "0",
        "producer": {
            "kind": "preprocessing",
            "definition": {
                "definition_id": "pre",
                "version": "v1",
                "steps": [
                    {
                        "instance_name": "step",
                        "component": {
                            "registered_name": "normalize_unicode",
                            "version": "0",
                            "settings": [],
                        },
                    }
                ],
            },
        },
        "metrics_definition": {
            "definition_id": "def",
            "version": "1",
            "questions": [
                {
                    "metric": "code_leakage",
                    "on_key": "output",
                    "settings": [
                        {"name": "task_names", "value": ["add_one"]},
                    ],
                }
            ],
        },
    },
    "facts": [
        {"name": "keyword_count", "value": 2, "unit": "count"},
        {
            "name": "punctuation_density",
            "value": 0.25,
            "unit": "ratio",
        },
    ],
}


def _golden_record():
    from dr_code.metrics import MetricFactUnit, MetricName
    from dr_code.trace import ComponentSetting

    question = _question_coordinate(
        metric=MetricName.CODE_LEAKAGE,
        on_key="output",
        settings=(ComponentSetting(name="task_names", value=("add_one",)),),
    )
    return _measured(
        identity=_identity(question=question, metric_version="0"),
        facts=(
            _fact(name="keyword_count", value=2),
            _fact(
                name="punctuation_density",
                value=0.25,
                unit=MetricFactUnit.RATIO,
            ),
        ),
    )


def test_measured_record_serializes_to_the_golden_literals() -> None:
    assert (
        json.loads(_golden_record().model_dump_json())
        == _GOLDEN_MEASURED_RECORD
    )


def test_golden_literals_load_back_to_an_equal_record() -> None:
    from dr_code.metrics import METRIC_RECORD_ADAPTER

    restored = METRIC_RECORD_ADAPTER.validate_python(_GOLDEN_MEASURED_RECORD)
    assert restored == _golden_record()


_GOLDEN_NOT_APPLICABLE_RECORD = {
    "schema_version": 1,
    "status": "not_applicable",
    "identity": _GOLDEN_MEASURED_RECORD["identity"],
    "absence": {
        "kind": "absent",
        "failed_step": "extract",
        "failure_code": "no_candidates_extracted",
        "cause": "no code extracted",
        "propagated_through": ["clean"],
    },
}


_GOLDEN_OPERATOR_FAILURE_RECORD = {
    "schema_version": 1,
    "status": "operator_failure",
    "identity": _GOLDEN_MEASURED_RECORD["identity"],
    "failure": {
        "failure_type": "ValueError",
        "failure_message": "boom",
    },
}


def _golden_not_applicable():
    return _not_applicable(identity=_golden_record().identity)


def _golden_operator_failure():
    return _operator_failure(identity=_golden_record().identity)


def test_not_applicable_record_serializes_to_the_golden_literals() -> None:
    assert (
        json.loads(_golden_not_applicable().model_dump_json())
        == _GOLDEN_NOT_APPLICABLE_RECORD
    )


def test_golden_not_applicable_literals_load_back_equal() -> None:
    from dr_code.metrics import METRIC_RECORD_ADAPTER

    restored = METRIC_RECORD_ADAPTER.validate_python(
        _GOLDEN_NOT_APPLICABLE_RECORD
    )
    assert restored == _golden_not_applicable()


def test_operator_failure_record_serializes_to_the_golden_literals() -> None:
    assert (
        json.loads(_golden_operator_failure().model_dump_json())
        == _GOLDEN_OPERATOR_FAILURE_RECORD
    )


def test_golden_operator_failure_literals_load_back_equal() -> None:
    from dr_code.metrics import METRIC_RECORD_ADAPTER

    restored = METRIC_RECORD_ADAPTER.validate_python(
        _GOLDEN_OPERATOR_FAILURE_RECORD
    )
    assert restored == _golden_operator_failure()


class _PoisonedRegistry:
    def __getitem__(self, key: object) -> object:
        raise AssertionError(
            f"record deserialization consulted the registry: [{key!r}]"
        )

    def get(self, key: object, default: object = None) -> object:
        raise AssertionError(
            f"record deserialization consulted the registry: get({key!r})"
        )

    def __contains__(self, key: object) -> bool:
        raise AssertionError(
            f"record deserialization consulted the registry: {key!r} in ..."
        )

    def __iter__(self) -> object:
        raise AssertionError("record deserialization iterated the registry")


def test_record_loads_when_its_metric_is_absent_from_the_live_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.metrics import METRIC_RECORD_ADAPTER
    import dr_code.metrics.registry as registry_module

    payload = json.dumps(_GOLDEN_MEASURED_RECORD)
    monkeypatch.setattr(
        registry_module, "REGISTRY", _PoisonedRegistry(), raising=True
    )

    restored = METRIC_RECORD_ADAPTER.validate_json(payload)
    assert restored.identity.question.metric.value == "code_leakage"
    assert restored.identity.question.settings[0].name == "task_names"
    assert restored.identity.question.settings[0].value == ("add_one",)


def test_record_loads_settings_the_live_operator_no_longer_accepts() -> None:
    from dr_code.metrics import METRIC_RECORD_ADAPTER

    payload = json.loads(json.dumps(_GOLDEN_MEASURED_RECORD))
    retired = [{"name": "retired_setting", "value": 7}]
    payload["identity"]["question"]["settings"] = retired
    payload["identity"]["metrics_definition"]["questions"][0]["settings"] = (
        retired
    )

    restored = METRIC_RECORD_ADAPTER.validate_python(payload)
    assert restored.identity.question.settings[0].name == "retired_setting"
    assert restored.identity.question.settings[0].value == 7


def test_engine_produced_records_round_trip(text_trace) -> None:
    from dr_code.metrics import METRIC_RECORD_ADAPTER

    for record in _engine_records(text_trace):
        assert (
            METRIC_RECORD_ADAPTER.validate_json(record.model_dump_json())
            == record
        )


def _engine_records(text_trace) -> tuple[object, ...]:
    from dr_code.metrics import MetricName, MetricQuestion, MetricsDefinition
    from dr_code.metrics.engine.engine import extract_metrics

    definition = MetricsDefinition(
        definition_id="round-trip",
        version="1",
        questions=(
            MetricQuestion(
                metric=MetricName.COMPRESSED_LENGTH,
                on="output",
                settings={"compression": {"method": "gzip", "level": 9}},
            ),
            MetricQuestion(
                metric=MetricName.CODE_LEAKAGE,
                on="output",
                settings={"task_names": ["add_one"]},
            ),
        ),
    )
    return extract_metrics(definition, text_trace("hello world"))


@pytest.mark.parametrize(
    "metric_value,settings",
    [
        pytest.param(
            "compressed_length",
            {"compression": {"method": "gzip", "level": 9}},
            id="compressed_length-gzip",
        ),
        pytest.param(
            "compressed_length",
            {"compression": {"method": "zstd", "level": 3}},
            id="compressed_length-zstd",
        ),
        pytest.param(
            "code_leakage",
            {"task_names": ["alpha", "beta"]},
            id="code_leakage-task-names",
        ),
    ],
)
def test_every_status_shape_round_trips_with_non_trivial_settings(
    metric_value: str, settings: dict[str, object]
) -> None:
    from dr_code.metrics import (
        METRIC_RECORD_ADAPTER,
        MetricName,
        MetricQuestion,
        MetricQuestionCoordinate,
    )

    question = MetricQuestionCoordinate.of(
        MetricQuestion(
            metric=MetricName(metric_value), on="input", settings=settings
        )
    )
    identity = _identity(question=question)
    for record in (
        _measured(identity=identity),
        _not_applicable(identity=identity),
        _operator_failure(identity=identity),
    ):
        restored = METRIC_RECORD_ADAPTER.validate_json(
            record.model_dump_json()
        )
        assert restored == record
        assert restored.identity.question.settings == question.settings
