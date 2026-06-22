"""Unit tests for live worker tuning helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dr_code.pipeline import tune


def test_count_terminals_uses_runtime_status(monkeypatch) -> None:
    calls = []

    def get_run_status(run_id, *, run_store=None):
        calls.append((run_id, run_store))
        return SimpleNamespace(terminal_jobs=7, stages=[])

    monkeypatch.setattr(tune, "get_run_status", get_run_status)

    assert tune.count_terminals("run-1") == 7
    assert calls == [("run-1", None)]


def test_count_stage_completions_uses_runtime_status(monkeypatch) -> None:
    monkeypatch.setattr(
        tune,
        "get_run_status",
        lambda run_id, *, run_store=None: SimpleNamespace(
            stages=[
                SimpleNamespace(stage="parse", completed_jobs=5),
                SimpleNamespace(stage="test", completed_jobs=3),
            ],
        ),
    )

    assert tune.count_stage_completions("run-1", "parse") == 5


def test_count_stage_completions_rejects_unknown_stage(monkeypatch) -> None:
    monkeypatch.setattr(
        tune,
        "get_run_status",
        lambda run_id, *, run_store=None: SimpleNamespace(stages=[]),
    )

    with pytest.raises(ValueError, match="Unknown stage"):
        tune.count_stage_completions("run-1", "parse")
