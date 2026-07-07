"""Basic tests for inverse transforms."""

from __future__ import annotations

import random

import pytest

from code_eval.synthetic.equivalence import canonicalize
from code_eval.synthetic.humaneval_loader import load_humaneval_plus
from code_eval.synthetic.inverse_transforms import REGISTRY

# Pick a single small task for transform unit tests — keeps tests fast.
TARGET_TASK_ID = "HumanEval/0"


@pytest.fixture(scope="module")
def ground_truth() -> str:
    tasks = load_humaneval_plus(prefer_snapshot=True)
    by_id = {t.task_id: t for t in tasks}
    return canonicalize(by_id[TARGET_TASK_ID].full_source)


@pytest.mark.parametrize("transform_name", sorted(REGISTRY.keys()))
def test_transform_declares_recovery_steps(transform_name: str, ground_truth: str) -> None:
    transform_cls = REGISTRY[transform_name]
    transform = transform_cls()
    sample = transform.apply(ground_truth, random.Random(0xC0DE))

    assert (
        sample.expected_recovery_steps
    ), f"{transform_name}: declared empty recovery — transforms must declare at least one step."


@pytest.mark.parametrize("transform_name", sorted(REGISTRY.keys()))
def test_transform_is_deterministic(transform_name: str, ground_truth: str) -> None:
    """Same source + same rng state -> same output."""
    transform_cls = REGISTRY[transform_name]
    transform = transform_cls()
    a = transform.apply(ground_truth, random.Random(42))
    b = transform.apply(ground_truth, random.Random(42))
    assert a.corrupted_source == b.corrupted_source
    assert a.expected_recovery_steps == b.expected_recovery_steps
