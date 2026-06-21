"""Unit tests for HumanEval+ loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from dr_code.datasets.humaneval_loader import (
    get_task,
    load_humaneval_plus,
    task_index,
)


def test_load_humaneval_plus_from_snapshot() -> None:
    tasks = load_humaneval_plus(prefer_snapshot=True)
    assert len(tasks) == 164


def test_all_tasks_have_non_empty_test() -> None:
    tasks = load_humaneval_plus(prefer_snapshot=True)
    for task in tasks:
        assert task.test.strip()
        assert task.prompt.strip()
        assert task.entry_point.strip()


def test_get_task_and_index() -> None:
    task = get_task("HumanEval/0", prefer_snapshot=True)
    index = task_index(prefer_snapshot=True)
    assert task.task_id == "HumanEval/0"
    assert index["HumanEval/0"].entry_point == task.entry_point


def test_full_source_property() -> None:
    task = get_task("HumanEval/0", prefer_snapshot=True)
    assert task.full_source == task.prompt + task.canonical_solution


def test_snapshot_file_exists() -> None:
    snap = Path("tests/corpus/humanevalplus_snapshot.json")
    assert snap.exists()


def test_get_task_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown HumanEval\\+ task_id"):
        get_task("HumanEval/9999", prefer_snapshot=True)
