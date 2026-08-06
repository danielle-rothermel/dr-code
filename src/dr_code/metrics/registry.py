from collections.abc import Mapping
from types import MappingProxyType

from dr_code.metrics.operators.ast_stats import AstStats
from dr_code.metrics.operators.base import MetricOperator
from dr_code.metrics.operators.code_leakage import CodeLeakage
from dr_code.humaneval.metric_operator import CodeTest
from dr_code.metrics.operators.compressed_length import CompressedLength
from dr_code.metrics.operators.parse_outcome import ParseOutcome
from dr_code.metrics.operators.text_stats import TextStats

REGISTRY: Mapping[str, type[MetricOperator]] = MappingProxyType(
    {
        str(operator.NAME): operator
        for operator in (
            TextStats,
            CodeLeakage,
            ParseOutcome,
            AstStats,
            CompressedLength,
            CodeTest,
        )
    }
)
