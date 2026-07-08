"""Tests for HumanEvalPlus source selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from dr_code.synthetic import humaneval_loader
from dr_code.synthetic.humaneval_loader import HumanEvalPlusTask, load_humaneval_plus


def _task() -> HumanEvalPlusTask:
    return HumanEvalPlusTask(
        task_id="HumanEval/0",
        prompt="def add(a, b):\n",
        canonical_solution="    return a + b\n",
        entry_point="add",
        test="",
    )


def test_hf_first_does_not_silently_use_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_tasks = [_task()]

    def unavailable_hf() -> list[HumanEvalPlusTask]:
        raise FileNotFoundError("hf unavailable")

    def available_snapshot(_repo_root: Path) -> list[HumanEvalPlusTask]:
        return snapshot_tasks

    monkeypatch.setattr(humaneval_loader, "_load_from_hf", unavailable_hf)
    monkeypatch.setattr(humaneval_loader, "_load_from_snapshot", available_snapshot)

    with pytest.raises(FileNotFoundError, match="hf unavailable"):
        load_humaneval_plus(prefer_snapshot=False)


def test_snapshot_first_allows_explicit_offline_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_tasks = [_task()]

    def unavailable_hf() -> list[HumanEvalPlusTask]:
        raise FileNotFoundError("hf unavailable")

    def available_snapshot(_repo_root: Path) -> list[HumanEvalPlusTask]:
        return snapshot_tasks

    monkeypatch.setattr(humaneval_loader, "_load_from_hf", unavailable_hf)
    monkeypatch.setattr(humaneval_loader, "_load_from_snapshot", available_snapshot)

    assert load_humaneval_plus(prefer_snapshot=True) == snapshot_tasks
