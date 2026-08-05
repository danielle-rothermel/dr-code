"""Tests for HumanEval row snapshots and sampling."""

from __future__ import annotations

from pathlib import Path

import pytest

from _humaneval_builders import _row
from dr_code.humaneval import HumanEvalTask, parse_humaneval_dataset
from dr_code.humaneval.runner import require_parsed_tests
from dr_code.humaneval.sampling import (
    HumanEvalRawRowsSnapshot,
    load_humaneval_rows,
    sample_humaneval_tasks_from_rows,
    validate_snapshot_header,
)
from dr_code.humaneval.task import HUMANEVAL_OVERRIDE_SET


SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "corpus"
    / "humanevalplus_snapshot.json"
)


@pytest.fixture(scope="module")
def raw_snapshot() -> HumanEvalRawRowsSnapshot:
    return HumanEvalRawRowsSnapshot.model_validate_json(
        SNAPSHOT_PATH.read_text(encoding="utf-8")
    )


def _check_payload_bytes(task: HumanEvalTask) -> list[bytes]:
    parsed_tests = require_parsed_tests(task)
    return [
        case.as_check(
            candidate_name="candidate",
            assertion_name=parsed_tests.assertion_name,
        )
        .model_dump_json()
        .encode("utf-8")
        for case in parsed_tests.cases
    ]


def test_sampling_from_rows_is_deterministic_and_indexed() -> None:
    rows = [_row(f"HumanEval/{index}", index) for index in range(5)]

    first = sample_humaneval_tasks_from_rows(
        rows,
        seed=17,
        sample_count=3,
    )
    second = sample_humaneval_tasks_from_rows(
        rows,
        seed=17,
        sample_count=3,
    )

    assert [sample.sample_index for sample in first] == [0, 1, 2]
    assert [sample.task.task_id for sample in first] == [
        sample.task.task_id for sample in second
    ]


def test_raw_row_snapshot_rehydrates_byte_equal_checks(
    raw_snapshot: HumanEvalRawRowsSnapshot,
) -> None:
    snapshot_rows = [row.model_dump(mode="json") for row in raw_snapshot.rows]
    loaded_rows = load_humaneval_rows(snapshot_path=SNAPSHOT_PATH)

    assert raw_snapshot.header.override_set == HUMANEVAL_OVERRIDE_SET
    assert loaded_rows == snapshot_rows

    tasks = parse_humaneval_dataset(loaded_rows)
    assert [task.task_id for task in tasks] == [
        row.task_id for row in raw_snapshot.rows
    ]

    override_entry = HUMANEVAL_OVERRIDE_SET.entries[0]
    override_task = next(
        task for task in tasks if task.task_id == override_entry.task_id
    )
    assert override_task.notes == override_entry.override.notes
    assert (
        override_task.canonical_solution
        == override_entry.override.canonical_solution
    )
    for replacement in override_entry.override.test_replacements:
        assert replacement.old not in override_task.test
        assert replacement.replacement in override_task.test

    for task in tasks:
        generated_checks = _check_payload_bytes(task)
        assert generated_checks
        assert all(generated_checks)


@pytest.mark.parametrize(
    "mismatch",
    (
        "dataset",
        "revision",
        "override-id",
        "override-version",
        "override-structure",
    ),
)
def test_raw_row_snapshot_rejects_provenance_mismatch(
    raw_snapshot: HumanEvalRawRowsSnapshot,
    mismatch: str,
) -> None:
    header = raw_snapshot.header
    if mismatch == "dataset":
        header = header.model_copy(update={"dataset_id": "other/dataset"})
        expected = (
            "HumanEval raw-row snapshot dataset mismatch: "
            f"{header.dataset_id!r} != {raw_snapshot.header.dataset_id!r}"
        )
    elif mismatch == "revision":
        header = header.model_copy(update={"hf_revision": "other-revision"})
        expected = (
            "HumanEval raw-row snapshot HF revision mismatch: "
            f"{header.hf_revision!r} != {raw_snapshot.header.hf_revision!r}"
        )
    elif mismatch == "override-id":
        header = header.model_copy(
            update={
                "override_set": header.override_set.model_copy(
                    update={"override_set_id": "other-overrides"}
                )
            }
        )
        expected = (
            "unsupported HumanEval override set: "
            f"other-overrides@{header.override_set.version}"
        )
    elif mismatch == "override-version":
        header = header.model_copy(
            update={
                "override_set": header.override_set.model_copy(
                    update={"version": "other-version"}
                )
            }
        )
        expected = (
            "unsupported HumanEval override set: "
            f"{header.override_set.override_set_id}@other-version"
        )
    else:
        assert mismatch == "override-structure"
        header = header.model_copy(
            update={
                "override_set": header.override_set.model_copy(
                    update={"entries": ()}
                )
            }
        )
        expected = (
            "HumanEval raw-row snapshot override-set mismatch: "
            f"{header.override_set!r} != {HUMANEVAL_OVERRIDE_SET!r}"
        )

    with pytest.raises(ValueError) as exc_info:
        validate_snapshot_header(
            header,
            dataset_name=raw_snapshot.header.dataset_id,
            hf_revision=raw_snapshot.header.hf_revision,
        )

    assert str(exc_info.value) == expected
