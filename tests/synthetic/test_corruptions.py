"""Tests for synthetic corruption categories."""

from __future__ import annotations

import random

import pytest

from dr_code.code_transforms import canonicalize
from dr_code.synthetic.humaneval_loader import load_humaneval_plus
from dr_code.synthetic.corruptions import REGISTRY
from dr_code.synthetic.names import CorruptionName

TARGET_TASK_ID = "HumanEval/0"

TRANSFORM_CATEGORIES: dict[str, tuple[CorruptionName, ...]] = {
    "wrappers": (
        CorruptionName.ADD_CODE_FENCES,
        CorruptionName.ADD_PROSE_WRAPPER,
        CorruptionName.ADD_MARKDOWN_WRAPPERS,
        CorruptionName.ADD_INLINE_BACKTICKS,
        CorruptionName.ADD_MULTIPLE_SOLUTIONS,
    ),
    "whitespace_text": (
        CorruptionName.ADD_INDENTATION,
        CorruptionName.ADD_TABS,
        CorruptionName.ADD_TRAILING_WHITESPACE,
        CorruptionName.ADD_CRLF,
        CorruptionName.ADD_UNICODE_NOISE,
        CorruptionName.ADD_BLANK_LINES,
    ),
    "imports": (
        CorruptionName.REMOVE_IMPORTS,
        CorruptionName.MANGLE_IMPORT_LINES,
        CorruptionName.DUPLICATE_IMPORTS,
    ),
    "syntax": (CorruptionName.TRUNCATE,),
    "normalization_noise": (
        CorruptionName.ADD_SMART_QUOTES,
        CorruptionName.ADD_COMMENTS_NOISE,
        CorruptionName.ADD_DEAD_CODE,
        CorruptionName.CHANGE_QUOTE_STYLE,
        CorruptionName.CHANGE_STRING_FORM,
        CorruptionName.ADD_TYPE_ANNOTATIONS,
        CorruptionName.RENAME_LOCALS,
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
    transform_names: tuple[CorruptionName, ...],
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
    assert set(REGISTRY) == {transform.value for transform in CorruptionName}
