"""Stage 4 analysis: compression joins and aggregates."""

from dr_code.analysis.aggregate import (
    aggregate_by_compression_quartile,
    aggregate_by_model,
    aggregate_by_source,
    aggregate_by_task,
    build_aggregates,
    build_summary,
)
from dr_code.analysis.compress import decoder_input_compression
from dr_code.analysis.export import AnalysisArtifacts, export_analysis
from dr_code.analysis.join import (
    EnrichedRow,
    JoinReport,
    enrich_eval_run,
    load_attempts,
    load_parse_outcomes,
    load_test_outcomes,
)

__all__ = [
    "AnalysisArtifacts",
    "EnrichedRow",
    "JoinReport",
    "aggregate_by_compression_quartile",
    "aggregate_by_model",
    "aggregate_by_source",
    "aggregate_by_task",
    "build_aggregates",
    "build_summary",
    "decoder_input_compression",
    "enrich_eval_run",
    "export_analysis",
    "load_attempts",
    "load_parse_outcomes",
    "load_test_outcomes",
]
