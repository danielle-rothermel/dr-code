from __future__ import annotations


def _producer():
    from dr_code.trace import (
        ComponentCoordinate,
        PreprocessingDefinitionCoordinate,
        PreprocessingTraceProducer,
        StepCoordinate,
    )

    return PreprocessingTraceProducer(
        definition=PreprocessingDefinitionCoordinate(
            definition_id="pre",
            version="v1",
            steps=(
                StepCoordinate(
                    instance_name="step",
                    component=ComponentCoordinate(
                        registered_name="normalize_unicode",
                        version="0",
                    ),
                ),
            ),
        )
    )


def _question_coordinate(metric=None, on_key="input", settings=()):
    from dr_code.metrics import MetricName, MetricQuestionCoordinate

    return MetricQuestionCoordinate(
        metric=MetricName.TEXT_STATS if metric is None else metric,
        on_key=on_key,
        settings=settings,
    )


def _identity(question=None, **overrides):
    from dr_code.metrics import (
        MetricRecordId,
        MetricsDefinitionCoordinate,
    )

    question = question if question is not None else _question_coordinate()
    base: dict[str, object] = {
        "question": question,
        "metric_version": "1",
        "producer": _producer(),
        "metrics_definition": MetricsDefinitionCoordinate(
            definition_id="def",
            version="1",
            questions=(question,),
        ),
    }
    base.update(overrides)
    return MetricRecordId(**base)


def _value(name="character_count", value=4, unit=None):
    from dr_code.metrics import MetricValue, MetricValueUnit

    return MetricValue(
        name=name,
        value=value,
        unit=MetricValueUnit.COUNT if unit is None else unit,
    )


def _measured(**overrides):
    from dr_code.metrics import MeasuredRecord

    base: dict[str, object] = {
        "identity": _identity(),
        "values": (_value(),),
    }
    base.update(overrides)
    return MeasuredRecord(**base)


def _absent():
    from dr_code.trace import Absent

    return Absent(
        failed_step="extract",
        failure_code="no_candidates_extracted",
        cause="no code extracted",
        propagated_through=("clean",),
    )


def _not_applicable(**overrides):
    from dr_code.metrics import NotApplicableRecord

    base: dict[str, object] = {
        "identity": _identity(),
        "absence": _absent(),
    }
    base.update(overrides)
    return NotApplicableRecord(**base)


def _operator_failure(**overrides):
    from dr_code.metrics import OperatorFailure, OperatorFailureRecord

    base: dict[str, object] = {
        "identity": _identity(),
        "failure": OperatorFailure(
            failure_type="ValueError", failure_message="boom"
        ),
    }
    base.update(overrides)
    return OperatorFailureRecord(**base)
