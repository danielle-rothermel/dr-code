from typing import TYPE_CHECKING

from dr_code.metrics.coordinates import (
    MetricQuestionCoordinate,
    MetricValueCoordinate,
    MetricsDefinitionCoordinate,
)
from dr_code.metrics.definition import (
    MetricQuestion,
    MetricsDefinition,
)
from dr_code.metrics.names import MetricName
from dr_code.metrics.records import (
    METRIC_RECORD_ADAPTER,
    METRIC_RECORD_SCHEMA_VERSION,
    MeasuredRecord,
    MetricValue,
    MetricRecord,
    MetricRecordIdentity,
    NotApplicableRecord,
    OperatorFailure,
    OperatorFailureRecord,
    RecordStatus,
    record_rows,
)
from dr_code.metrics.settings import OperatorSettings
from dr_code.metrics.units import MetricValueUnit

if TYPE_CHECKING:
    from dr_code.metrics.engine.engine import (
        EngineInvariantError,
        extract_metrics,
        extract_metrics_batch,
    )

_ENGINE_EXPORTS = frozenset(
    {
        "EngineInvariantError",
        "extract_metrics",
        "extract_metrics_batch",
    }
)


def __getattr__(name: str) -> object:
    if name not in _ENGINE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from dr_code.metrics.engine import engine

    value = getattr(engine, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _ENGINE_EXPORTS)


__all__ = (
    "METRIC_RECORD_ADAPTER",
    "METRIC_RECORD_SCHEMA_VERSION",
    "MeasuredRecord",
    "MetricValue",
    "MetricValueCoordinate",
    "MetricValueUnit",
    "MetricName",
    "MetricQuestion",
    "MetricQuestionCoordinate",
    "MetricRecord",
    "MetricRecordIdentity",
    "MetricsDefinition",
    "MetricsDefinitionCoordinate",
    "NotApplicableRecord",
    "OperatorFailure",
    "OperatorFailureRecord",
    "OperatorSettings",
    "RecordStatus",
    "EngineInvariantError",
    "extract_metrics",
    "extract_metrics_batch",
    "record_rows",
)
