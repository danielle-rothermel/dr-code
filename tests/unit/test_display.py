"""Unit tests for side-by-side spot-check display."""

from __future__ import annotations

import pytest

from dr_code.datasets.display import format_spot_check
from dr_code.datasets.humaneval_loader import get_task
from dr_code.models.attempts import AttemptRecord


def test_format_spot_check_includes_pool_and_fresh_blocks() -> None:
    task = get_task("HumanEval/0", prefer_snapshot=True)
    pool_records = [
        AttemptRecord.from_dedup_row(
            out="def has_close_elements(numbers, threshold):\n    pass\n",
            count=3,
            task_id=task.task_id,
            decoder_input="compressed description",
        )
    ]
    fresh_records = [
        AttemptRecord.stub_for_fresh(
            task,
            decoder_input=task.prompt,
            raw_output="def has_close_elements(numbers, threshold):\n    return False\n",
            run_id="demo-run",
            model="google/gemini-3.1-flash-lite",
            profile_id="openrouter/google/gemini-3.1-flash-lite/off/v1",
        )
    ]
    rendered = format_spot_check(
        pool_records,
        fresh_records,
        task_id=task.task_id,
    )
    assert "Spot check: HumanEval/0" in rendered
    assert "[pool]" in rendered
    assert "[fresh_stub]" in rendered
    assert "occurrence_count: 3" in rendered
    assert "profile: openrouter/google/gemini-3.1-flash-lite/off/v1" in rendered


def test_format_spot_check_missing_task_raises() -> None:
    task = get_task("HumanEval/0", prefer_snapshot=True)
    pool_records = [
        AttemptRecord.from_dedup_row(
            out="def has_close_elements(numbers, threshold):\n    pass\n",
            count=1,
            task_id=task.task_id,
            decoder_input="compressed description",
        )
    ]
    with pytest.raises(ValueError, match="No fresh_stub records"):
        format_spot_check(pool_records, [], task_id=task.task_id)
