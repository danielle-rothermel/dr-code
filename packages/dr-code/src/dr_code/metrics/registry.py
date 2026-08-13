from collections.abc import Mapping
from importlib.metadata import entry_points
from types import MappingProxyType
from typing import overload

from dr_code.metrics.operators.ast_stats import AstStats
from dr_code.metrics.operators.base import MetricOperator
from dr_code.metrics.operators.code_leakage import CodeLeakage
from dr_code.metrics.operators.compressed_length import CompressedLength
from dr_code.metrics.operators.parse_outcome import ParseOutcome
from dr_code.metrics.operators.text_stats import TextStats

_BUILTIN_OPERATORS: tuple[type[MetricOperator], ...] = (
    TextStats,
    CodeLeakage,
    ParseOutcome,
    AstStats,
    CompressedLength,
)

_METRIC_OPERATOR_GROUP = "dr_code.metric_operators"
_REGISTRY: Mapping[str, type[MetricOperator]] | None = None


def _load_registry() -> Mapping[str, type[MetricOperator]]:
    operators: dict[str, type[MetricOperator]] = {
        str(operator.NAME): operator for operator in _BUILTIN_OPERATORS
    }
    for entry_point in entry_points(group=_METRIC_OPERATOR_GROUP):
        operator = entry_point.load()
        operators[entry_point.name] = operator
    return MappingProxyType(operators)


def _metric_registry() -> Mapping[str, type[MetricOperator]]:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _load_registry()
    return _REGISTRY


def register_metric_operator(
    name: str,
    operator: type[MetricOperator],
    /,
) -> None:
    global _REGISTRY
    updated = dict(_metric_registry())
    updated[name] = operator
    _REGISTRY = MappingProxyType(updated)


class _MetricRegistry(Mapping[str, type[MetricOperator]]):
    def __getitem__(self, key: str) -> type[MetricOperator]:
        return _metric_registry()[key]

    def __iter__(self):
        return iter(_metric_registry())

    def __len__(self) -> int:
        return len(_metric_registry())

    @overload
    def get(
        self, key: object, default: None = None
    ) -> type[MetricOperator] | None: ...

    @overload
    def get[T](
        self, key: object, default: T, /
    ) -> type[MetricOperator] | T: ...

    def get(self, key: object, default=None):
        if not isinstance(key, str):
            return default
        return _metric_registry().get(key, default)


REGISTRY = _MetricRegistry()
