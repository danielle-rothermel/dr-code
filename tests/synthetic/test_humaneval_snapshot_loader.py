"""Tests for HumanEvalPlus source selection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from dr_code.synthetic import humaneval_loader
from dr_code.synthetic.humaneval_loader import load_humaneval_plus


ROW = {
    "task_id": "HumanEval/0",
    "prompt": "def add(a, b):\n",
    "canonical_solution": "    return a + b\n",
    "entry_point": "add",
    "test": "",
}


def test_default_loader_does_not_fall_back_to_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Mapping[str, object]] = []

    def unavailable_source(**kwargs: object) -> list[object]:
        calls.append(kwargs)
        raise FileNotFoundError("hf unavailable")

    monkeypatch.setattr(
        humaneval_loader,
        "load_human_eval_rows",
        unavailable_source,
    )

    with pytest.raises(FileNotFoundError, match="hf unavailable"):
        load_humaneval_plus(prefer_snapshot=False)

    assert len(calls) == 1
    assert "snapshot_path" not in calls[0]


def test_explicit_snapshot_loader_uses_repository_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Mapping[str, object]] = []

    def available_source(**kwargs: object) -> list[dict[str, str]]:
        calls.append(kwargs)
        return [ROW]

    monkeypatch.setattr(
        humaneval_loader,
        "load_human_eval_rows",
        available_source,
    )

    tasks = load_humaneval_plus(prefer_snapshot=True)

    assert [task.task_id for task in tasks] == ["HumanEval/0"]
    assert calls[0]["snapshot_path"] == (
        Path(__file__).resolve().parents[2]
        / humaneval_loader.SNAPSHOT_REL_PATH
    )
