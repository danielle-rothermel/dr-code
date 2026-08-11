from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from _humaneval_builders import _input_result_test, _row, _task
from dr_code.humaneval import HumanEvalTask, parse_humaneval_dataset
from dr_code.humaneval.metric_operator import _validate_task_payload
from dr_code.humaneval.parsed_code import parse_code
from dr_code.humaneval.parsed_tests import (
    InputResultTestCase,
    parse_humaneval_tests,
)
from dr_code.humaneval.task import (
    HumanEvalOverride,
    HumanEvalTestReplacement,
    _apply_humaneval_override,
)
from dr_code.trace import JsonArtifact


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


def test_task_derives_its_parses_from_its_source_and_test_fields() -> None:
    task = _task()

    assert task.parsed == parse_code(
        display_title=task.task_id,
        code_str=task.ground_truth_code,
    )
    assert task.parsed_tests == parse_humaneval_tests(task.test)


def test_task_rejects_supplied_parses() -> None:
    other = _task(
        test=(
            "def check(candidate):\n"
            "    inputs = [(9,)]\n"
            "    results = [10]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n"
        )
    )

    with pytest.raises(ValidationError, match="parsed_tests"):
        HumanEvalTask(
            task_id="HumanEval/fixture",
            prompt="def add_one(x):\n",
            canonical_solution="    return x + 1\n",
            entry_point="add_one",
            test=_input_result_test(),
            parsed_tests=other.parsed_tests,  # type: ignore[call-arg]
        )


def test_task_is_frozen_and_its_notes_are_immutable() -> None:
    task = _task()

    assert isinstance(task.notes, tuple)
    with pytest.raises(ValidationError):
        task.notes = ("added later",)  # type: ignore[misc]
    with pytest.raises(ValidationError):
        task.prompt = "def other(x):\n"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        task.notes.append("added later")  # type: ignore[attr-defined]


def test_task_round_trips_through_its_field_payload() -> None:
    task = _task(test=_input_result_test())
    payload = json.loads(
        task.model_dump_json(exclude=set(HumanEvalTask.model_computed_fields))
    )

    restored = HumanEvalTask.model_validate(payload)

    assert restored == task
    assert restored.parsed == task.parsed
    assert restored.parsed_tests == task.parsed_tests


_TUPLE_EXPECTATION_TEST = (
    "def check(candidate):\n"
    "    inputs = [(1,), (2,)]\n"
    "    results = [(0, 1), (1, 2)]\n"
    "    for inp, exp in zip(inputs, results):\n"
    "        assertion(candidate(*inp), exp, 0)\n"
)

_NESTED_TUPLE_EXPECTATION_TEST = (
    "def check(candidate):\n"
    "    inputs = [(1,), (2,)]\n"
    "    results = [((0, 1), (2, 3)), ((4, 5), (6, 7))]\n"
    "    for inp, exp in zip(inputs, results):\n"
    "        assertion(candidate(*inp), exp, 0)\n"
)


@pytest.mark.parametrize(
    ("test", "expected_first"),
    [
        (_TUPLE_EXPECTATION_TEST, (0, 1)),
        (_NESTED_TUPLE_EXPECTATION_TEST, ((0, 1), (2, 3))),
    ],
    ids=["tuple_expectations", "nested_tuple_expectations"],
)
def test_tuple_expectations_survive_the_task_artifact_boundary(
    test: str,
    expected_first: object,
) -> None:
    task = _task(test=test)
    payload = json.loads(task.model_dump_json())

    restored = _validate_task_payload(JsonArtifact(payload=payload))

    cases = restored.parsed_tests.cases
    assert all(isinstance(case, InputResultTestCase) for case in cases)
    first = cases[0]
    assert isinstance(first, InputResultTestCase)
    assert first.expected == expected_first
    assert isinstance(first.expected, tuple)
    assert restored.parsed_tests == task.parsed_tests


def test_task_artifact_payload_omits_the_derived_parses() -> None:
    task = _task(test=_TUPLE_EXPECTATION_TEST)

    payload = json.loads(task.model_dump_json())

    assert "parsed_tests" not in payload
    assert "parsed" not in payload


def test_override_notes_reach_the_task_as_an_immutable_tuple() -> None:
    row = _row("HumanEval/99", 1)
    applied = _apply_humaneval_override(
        row,
        {"HumanEval/99": HumanEvalOverride(notes=("fixed the assertion",))},
    )
    task = HumanEvalTask(**applied)

    assert task.notes == ("fixed the assertion",)
