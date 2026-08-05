"""Tests for HumanEvalPlus source selection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from dr_code.humaneval.task import HUMANEVAL_OVERRIDE_SET
from dr_code.synthetic import humaneval_loader
from dr_code.synthetic.humaneval_loader import load_humaneval_plus


TEST_SOURCE = (
    "def check(candidate):\n"
    "    inputs = [[1, 2]]\n"
    "    results = [3]\n"
    "    for i, (inp, exp) in enumerate(zip(inputs, results)):\n"
    "        assert candidate(*inp) == exp\n"
)

ROW = {
    "task_id": "HumanEval/0",
    "prompt": "def add(a, b):\n",
    "canonical_solution": "    return a + b\n",
    "entry_point": "add",
    "test": TEST_SOURCE,
}

#: The one registered override entry that carries a test replacement; its
#: anchor is what a corrupt snapshot row can silently lose.
OVERRIDDEN_ENTRY = next(
    entry
    for entry in HUMANEVAL_OVERRIDE_SET.entries
    if entry.override.test_replacements
)


def test_default_loader_does_not_fall_back_to_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Mapping[str, object]] = []

    def unavailable_source(**kwargs: object) -> list[object]:
        calls.append(kwargs)
        raise FileNotFoundError("hf unavailable")

    monkeypatch.setattr(
        humaneval_loader,
        "load_humaneval_rows",
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
        "load_humaneval_rows",
        available_source,
    )

    tasks = load_humaneval_plus(prefer_snapshot=True)

    assert [task.task_id for task in tasks] == ["HumanEval/0"]
    assert calls[0]["snapshot_path"] == (
        Path(__file__).resolve().parents[2]
        / humaneval_loader.SNAPSHOT_REL_PATH
    )


@pytest.mark.parametrize("prefer_snapshot", [True, False])
def test_loader_rejects_row_missing_its_override_anchor(
    monkeypatch: pytest.MonkeyPatch, prefer_snapshot: bool
) -> None:
    # The snapshot loader guarantees provenance only, so the synthetic
    # builder is the use site that must reject a row whose overridden task
    # lost the text its registered replacement anchors on.
    corrupt_row = {
        **ROW,
        "task_id": OVERRIDDEN_ENTRY.task_id,
        "test": TEST_SOURCE,
    }

    def rows_with_corrupt_override(**_: object) -> list[dict[str, str]]:
        return [corrupt_row]

    monkeypatch.setattr(
        humaneval_loader,
        "load_humaneval_rows",
        rows_with_corrupt_override,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Override replacement text not found for "
            f"{OVERRIDDEN_ENTRY.task_id}"
        ),
    ):
        load_humaneval_plus(prefer_snapshot=prefer_snapshot)
