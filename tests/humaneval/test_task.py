"""Tests for HumanEval task construction and overrides."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from _humaneval_builders import _input_result_test, _row, _task
from dr_code.humaneval import HumanEvalTask, parse_humaneval_dataset
from dr_code.humaneval.task import (
    HumanEvalOverride,
    HumanEvalTestReplacement,
    _apply_humaneval_override,
)


def test_apply_humaneval_override_passthrough() -> None:
    row = _row("HumanEval/99", 1)
    assert _apply_humaneval_override(row, {}) == dict(row)

    updated = _apply_humaneval_override(
        row,
        {
            "HumanEval/99": HumanEvalOverride(
                canonical_solution="    return x + 99\n",
            ),
        },
    )
    assert updated["canonical_solution"] == "    return x + 99\n"

    with pytest.raises(ValueError, match="replacement text not found"):
        _apply_humaneval_override(
            row,
            {
                "HumanEval/99": HumanEvalOverride(
                    test_replacements=(
                        HumanEvalTestReplacement(
                            old="missing",
                            replacement="text",
                        ),
                    ),
                ),
            },
        )


def test_parse_humaneval_dataset_builds_tasks() -> None:
    tasks = parse_humaneval_dataset([_row("HumanEval/0", 0)])

    assert len(tasks) == 1
    assert tasks[0].task_id == "HumanEval/0"
    assert tasks[0].parsed_tests is not None


def test_task_recomputes_parses_instead_of_trusting_them() -> None:
    """A supplied parse is derived truth, not caller-supplied truth.

    Omitting the parses and supplying the correct ones produce the same task,
    so nothing is gained by supplying them -- and nothing wrong can be
    smuggled in by supplying them.
    """
    derived = _task()
    supplied = HumanEvalTask(
        task_id=derived.task_id,
        prompt=derived.prompt,
        canonical_solution=derived.canonical_solution,
        entry_point=derived.entry_point,
        test=derived.test,
        parsed=derived.parsed,
        parsed_tests=derived.parsed_tests,
    )

    assert supplied == derived


def test_task_rejects_a_parse_disagreeing_with_its_source() -> None:
    """A parse of some other code cannot ride along with this task."""
    other = HumanEvalTask(
        task_id="HumanEval/other",
        prompt="def subtract_one(x):\n",
        canonical_solution="    return x - 1\n",
        entry_point="subtract_one",
        test=_input_result_test(),
    )

    with pytest.raises(
        ValueError,
        match="parsed code must match prompt and canonical_solution",
    ):
        HumanEvalTask(
            task_id="HumanEval/fixture",
            prompt="def add_one(x):\n",
            canonical_solution="    return x + 1\n",
            entry_point="add_one",
            test=_input_result_test(),
            parsed=other.parsed,
        )


def test_task_rejects_parsed_tests_disagreeing_with_the_test_field() -> None:
    """Parsed tests cannot describe cases the raw ``test`` field does not."""
    other_test = (
        "def check(candidate):\n"
        "    inputs = [(9,)]\n"
        "    results = [10]\n"
        "    for inp, expected in zip(inputs, results):\n"
        "        assertion(candidate(*inp), expected)\n"
    )
    other = _task(test=other_test)

    with pytest.raises(
        ValueError,
        match="parsed tests must match the raw test field",
    ):
        HumanEvalTask(
            task_id="HumanEval/fixture",
            prompt="def add_one(x):\n",
            canonical_solution="    return x + 1\n",
            entry_point="add_one",
            test=_input_result_test(),
            parsed_tests=other.parsed_tests,
        )


def test_task_is_frozen_and_its_notes_are_immutable() -> None:
    """Nothing can edit a validated task after the fact."""
    task = _task()

    assert isinstance(task.notes, tuple)
    with pytest.raises(ValidationError):
        task.notes = ("added later",)  # type: ignore[misc]
    with pytest.raises(ValidationError):
        task.prompt = "def other(x):\n"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        task.notes.append("added later")  # type: ignore[attr-defined]


def test_task_round_trips_through_its_field_payload() -> None:
    """Serialized fields rebuild an equal task, parses included.

    Computed fields are excluded because they are derived, not stored: the
    field payload is the task's identity and the parses come back from it.
    """
    task = _task(test=_input_result_test())
    payload = json.loads(
        task.model_dump_json(exclude=set(HumanEvalTask.model_computed_fields))
    )

    restored = HumanEvalTask.model_validate(payload)

    assert restored == task
    assert restored.parsed == task.parsed
    assert restored.parsed_tests == task.parsed_tests


def test_override_notes_reach_the_task_as_an_immutable_tuple() -> None:
    """Override notes land on the frozen task without a mutable seam."""
    row = _row("HumanEval/99", 1)
    applied = _apply_humaneval_override(
        row,
        {"HumanEval/99": HumanEvalOverride(notes=("fixed the assertion",))},
    )
    task = HumanEvalTask(**applied)

    assert task.notes == ("fixed the assertion",)
