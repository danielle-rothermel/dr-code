"""Contracts for the mutant suite's test-only HumanEval+ cache."""

from __future__ import annotations

from pathlib import Path

from dr_code.synthetic.humaneval_loader import HumanEvalPlusTask
from mutants.humaneval_cache import (
    build_cached_loader,
    load_snapshot_tasks,
)


def _task(task_id: str) -> HumanEvalPlusTask:
    return HumanEvalPlusTask(
        task_id=task_id,
        prompt="def f():\n",
        canonical_solution="    return 1\n",
        entry_point="f",
        test="def check(candidate):\n    assert candidate() == 1\n",
    )


def test_cached_loader_returns_a_fresh_snapshot_list() -> None:
    snapshot_tasks = (_task("HumanEval/0"),)

    def unexpected_original_load(
        _prefer_snapshot: bool,
    ) -> list[HumanEvalPlusTask]:
        raise AssertionError("snapshot calls must use the cache")

    loader = build_cached_loader(
        snapshot_tasks=snapshot_tasks,
        original_loader=unexpected_original_load,
    )

    first = loader(True)
    second = loader(True)
    first.clear()

    assert first == []
    assert second == list(snapshot_tasks)


def test_cached_loader_delegates_non_snapshot_calls() -> None:
    delegated_task = _task("HumanEval/HF")
    calls: list[bool] = []

    def original_loader(
        prefer_snapshot: bool,
    ) -> list[HumanEvalPlusTask]:
        calls.append(prefer_snapshot)
        return [delegated_task]

    loader = build_cached_loader(
        snapshot_tasks=(_task("HumanEval/snapshot"),),
        original_loader=original_loader,
    )

    assert loader(False) == [delegated_task]
    assert calls == [False]


def test_shared_cache_serializes_once_and_reuses_tasks(tmp_path: Path) -> None:
    expected = (_task("HumanEval/0"), _task("HumanEval/1"))
    load_count = 0

    def original_loader(
        prefer_snapshot: bool,
    ) -> list[HumanEvalPlusTask]:
        nonlocal load_count
        assert prefer_snapshot is True
        load_count += 1
        return list(expected)

    first = load_snapshot_tasks(
        original_loader=original_loader,
        shared_temp_root=tmp_path,
    )
    second = load_snapshot_tasks(
        original_loader=original_loader,
        shared_temp_root=tmp_path,
    )

    assert load_count == 1
    assert first == expected
    assert second == expected
    assert second is not first
