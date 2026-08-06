from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def _load_bootstrap_helper():
    path = Path(__file__).parents[2] / "scripts" / "bootstrap_statistics.py"
    spec = spec_from_file_location("bootstrap_statistics_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bootstrap_confidence_interval = (
    _load_bootstrap_helper().bootstrap_confidence_interval
)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def test_bootstrap_interval_is_deterministic_and_contains_estimate() -> None:
    first = bootstrap_confidence_interval(
        [1.0, 2.0, 3.0, 4.0],
        _mean,
        confidence_level=0.90,
        resamples=1_000,
        seed=17,
    )
    second = bootstrap_confidence_interval(
        [1.0, 2.0, 3.0, 4.0],
        _mean,
        confidence_level=0.90,
        resamples=1_000,
        seed=17,
    )

    assert first == second
    assert first.estimate == 2.5
    assert first.lower <= first.estimate <= first.upper
    assert first.confidence_level == 0.90
    assert first.resamples == 1_000


@pytest.mark.parametrize(
    ("observations", "confidence_level", "resamples", "message"),
    [
        ([], 0.95, 100, "at least one observation"),
        ([1.0], 0.0, 100, "between zero and one"),
        ([1.0], 1.0, 100, "between zero and one"),
        ([1.0], 0.95, 0, "must be positive"),
    ],
)
def test_bootstrap_rejects_invalid_configuration(
    observations: list[float],
    confidence_level: float,
    resamples: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        bootstrap_confidence_interval(
            observations,
            _mean,
            confidence_level=confidence_level,
            resamples=resamples,
        )


def test_bootstrap_rejects_non_finite_statistics() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        bootstrap_confidence_interval([1.0], lambda _: math.inf, resamples=1)
