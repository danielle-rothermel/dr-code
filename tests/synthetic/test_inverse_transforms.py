"""Tests for synthetic inverse-transform categories."""

from __future__ import annotations

import random

import pytest

from dr_code.synthetic.equivalence import canonicalize
from dr_code.synthetic.humaneval_loader import load_humaneval_plus
from dr_code.synthetic.inverse_transforms import REGISTRY
from dr_code.synthetic.names import InverseTransformName

TARGET_TASK_ID = "HumanEval/0"

TRANSFORM_CATEGORIES: dict[str, tuple[InverseTransformName, ...]] = {
    "wrappers": (
        InverseTransformName.ADD_CODE_FENCES,
        InverseTransformName.ADD_PROSE_WRAPPER,
        InverseTransformName.ADD_MARKDOWN_WRAPPERS,
        InverseTransformName.ADD_INLINE_BACKTICKS,
        InverseTransformName.ADD_MULTIPLE_SOLUTIONS,
    ),
    "whitespace_text": (
        InverseTransformName.ADD_INDENTATION,
        InverseTransformName.ADD_TABS,
        InverseTransformName.ADD_TRAILING_WHITESPACE,
        InverseTransformName.ADD_CRLF,
        InverseTransformName.ADD_UNICODE_NOISE,
        InverseTransformName.ADD_BLANK_LINES,
    ),
    "imports": (
        InverseTransformName.REMOVE_IMPORTS,
        InverseTransformName.MANGLE_IMPORT_LINES,
        InverseTransformName.DUPLICATE_IMPORTS,
    ),
    "syntax": (InverseTransformName.TRUNCATE,),
    "normalization_noise": (
        InverseTransformName.ADD_SMART_QUOTES,
        InverseTransformName.ADD_COMMENTS_NOISE,
        InverseTransformName.ADD_DEAD_CODE,
        InverseTransformName.CHANGE_QUOTE_STYLE,
        InverseTransformName.CHANGE_STRING_FORM,
        InverseTransformName.ADD_TYPE_ANNOTATIONS,
        InverseTransformName.RENAME_LOCALS,
    ),
}


@pytest.fixture(scope="module")
def ground_truth() -> str:
    tasks = load_humaneval_plus(prefer_snapshot=True)
    by_id = {task.task_id: task for task in tasks}
    return canonicalize(by_id[TARGET_TASK_ID].full_source)


@pytest.mark.parametrize(
    ("category", "transform_names"),
    sorted(TRANSFORM_CATEGORIES.items()),
)
def test_transform_category_is_deterministic(
    category: str,
    transform_names: tuple[InverseTransformName, ...],
    ground_truth: str,
) -> None:
    assert transform_names, f"{category} category must include transforms"
    for transform_name in transform_names:
        transform_cls = REGISTRY[transform_name.value]
        transform = transform_cls()
        first = transform.apply(ground_truth, random.Random(42))
        second = transform.apply(ground_truth, random.Random(42))

        assert first == second
        assert isinstance(first.corrupted_source, str)
        assert not hasattr(first, "expected_recovery_steps")


def test_registry_covers_all_named_transforms() -> None:
    assert set(REGISTRY) == {transform.value for transform in InverseTransformName}
