"""Small reusable helpers for percentile bootstrap confidence intervals."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

ObservationT = TypeVar("ObservationT")


@dataclass(frozen=True, slots=True)
class BootstrapConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    resamples: int


def bootstrap_confidence_interval(
    observations: Sequence[ObservationT],
    statistic: Callable[[Sequence[ObservationT]], float],
    *,
    confidence_level: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapConfidenceInterval:
    """Return a deterministic percentile interval over resampled observations."""

    if not observations:
        raise ValueError("bootstrap requires at least one observation")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if resamples < 1:
        raise ValueError("resamples must be positive")

    frozen_observations = tuple(observations)
    estimate = _finite_statistic(statistic, frozen_observations)
    random_generator = random.Random(seed)
    sample_size = len(frozen_observations)
    bootstrap_values = sorted(
        _finite_statistic(
            statistic,
            tuple(
                frozen_observations[random_generator.randrange(sample_size)]
                for _ in range(sample_size)
            ),
        )
        for _ in range(resamples)
    )
    tail_probability = (1.0 - confidence_level) / 2.0
    return BootstrapConfidenceInterval(
        estimate=estimate,
        lower=_quantile(bootstrap_values, tail_probability),
        upper=_quantile(bootstrap_values, 1.0 - tail_probability),
        confidence_level=confidence_level,
        resamples=resamples,
    )


def _finite_statistic(
    statistic: Callable[[Sequence[ObservationT]], float],
    observations: Sequence[ObservationT],
) -> float:
    value = statistic(observations)
    if not math.isfinite(value):
        raise ValueError("bootstrap statistic must be finite")
    return value


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


__all__ = [
    "BootstrapConfidenceInterval",
    "bootstrap_confidence_interval",
]
