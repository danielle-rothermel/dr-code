"""Producer-blind metric operators and extraction.

``MetricQuestion`` and ``MetricsDefinition`` declare what to measure;
``extract_metrics`` answers a definition against a trace and returns one
``MetricRecord`` per declared question; ``record_facts`` projects a measured
record onto unit-carrying facts, and ``record_rows`` flattens records for
analysis.
"""

from dr_code.metrics.engine.engine import (
    EngineInvariantError,
    extract_metrics,
    extract_metrics_batch,
    record_facts,
)
from dr_code.metrics.definition import MetricQuestion, MetricsDefinition
from dr_code.metrics.names import MetricName
from dr_code.metrics.records import MetricRecord, RecordStatus, record_rows

__all__ = (
    "MetricName",
    "MetricQuestion",
    "MetricRecord",
    "MetricsDefinition",
    "EngineInvariantError",
    "RecordStatus",
    "extract_metrics",
    "extract_metrics_batch",
    "record_facts",
    "record_rows",
)
