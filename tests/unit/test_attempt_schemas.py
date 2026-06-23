"""Unit tests for AttemptRecord schemas."""

from __future__ import annotations

import pytest

from dr_code.models.attempts import (
    AttemptProvenance,
    AttemptRecord,
    AttemptSource,
    compute_sample_id,
)
from dr_code.models.humaneval import HumanEvalPlusTask


def test_compute_sample_id_is_stable() -> None:
    first = compute_sample_id("HumanEval/0", "print('hi')")
    second = compute_sample_id("HumanEval/0", "print('hi')")
    assert first == second
    assert len(first) == 16


def test_empty_decoder_input_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        AttemptRecord(
            sample_id="abc",
            run_id=None,
            task_id="HumanEval/0",
            decoder_input="",
            raw_output="x",
            provenance=AttemptProvenance(source=AttemptSource.POOL),
        )


def test_from_pool_row_maps_columns() -> None:
    row = {
        "human_eval_task_id": "HumanEval/0",
        "decoder_input_description": "desc",
        "raw_code_output": "code",
        "pool_name": "budget_dec_v0_size6",
        "model": "demo/model",
        "attempt_id": "attempt-1",
        "extra_col": "keep-me",
    }
    record = AttemptRecord.from_pool_row(row)
    assert record.task_id == "HumanEval/0"
    assert record.decoder_input == "desc"
    assert record.raw_output == "code"
    assert record.provenance.source is AttemptSource.POOL
    assert record.provenance.pool_name == "budget_dec_v0_size6"
    assert record.provenance.pool_attempt_id == "attempt-1"
    assert record.provenance.extra["extra_col"] == "keep-me"
    assert record.sample_id == compute_sample_id("HumanEval/0", "code")


def test_from_dedup_row_sets_occurrence_count() -> None:
    record = AttemptRecord.from_dedup_row(
        out="code",
        count=9,
        task_id="HumanEval/0",
        decoder_input="desc",
    )
    assert record.provenance.occurrence_count == 9


def test_stub_for_fresh() -> None:
    task = HumanEvalPlusTask(
        task_id="HumanEval/0",
        entry_point="has_close_elements",
        prompt="def has_close_elements(numbers, threshold):\n",
        canonical_solution="    return False",
        test="assert True",
    )
    record = AttemptRecord.stub_for_fresh(
        task,
        decoder_input=task.prompt,
        raw_output="def has_close_elements(numbers, threshold):\n    pass",
        run_id="run-1",
        model="demo/model",
    )
    assert record.run_id == "run-1"
    assert record.provenance.source is AttemptSource.FRESH_STUB
    assert record.provenance.model == "demo/model"
