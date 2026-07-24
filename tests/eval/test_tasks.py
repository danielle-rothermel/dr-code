"""HumanEval task, Task Set, and Repeat identity contracts."""

from __future__ import annotations

import pytest

from dr_code.eval import (
    RepeatId,
    RepeatPlan,
    RepeatProvenanceRow,
    TaskSet,
    humaneval_task_identity,
    repeat_plan_from_provenance,
)
from dr_code.humaneval.task import HumanEvalTask


def _task() -> HumanEvalTask:
    return HumanEvalTask(
        task_id="HumanEval/0",
        prompt="def f(x):\n",
        canonical_solution="    return x\n",
        entry_point="f",
        test=(
            "def check(candidate):\n"
            "    inputs = [(1,)]\n"
            "    results = [1]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assert candidate(*inp) == expected\n"
        ),
    )


def test_humaneval_identity_uses_dataset_fields_not_annotations() -> None:
    task = _task()
    annotated = task.model_copy(update={"notes": ["reviewed"]})
    assert humaneval_task_identity(task) == humaneval_task_identity(annotated)
    assert len(humaneval_task_identity(task)) == 64


def test_task_set_identity_preserves_manifest_order() -> None:
    forward = TaskSet(
        manifest_id="m",
        version="1",
        dataset_id="dataset",
        dataset_split="test",
        dataset_revision="r1",
        source_content_hash="a" * 64,
        source_task_identities=("a", "b"),
        task_identities=("a", "b"),
    )
    reverse = forward.model_copy(update={"task_identities": ("b", "a")})
    assert forward.identity_hash() != reverse.identity_hash()


def test_task_set_rejects_empty_manifest() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        TaskSet(
            manifest_id="m",
            version="1",
            dataset_id="dataset",
            dataset_split="test",
            dataset_revision="r1",
            source_content_hash="a" * 64,
            source_task_identities=("a",),
            task_identities=(),
        )


def test_task_set_rejects_selection_rule_payloads() -> None:
    with pytest.raises(ValueError, match="task_identities|selection_rule"):
        TaskSet.model_validate(
            {
                "manifest_id": "m",
                "version": "1",
                "dataset_id": "dataset",
                "dataset_split": "test",
                "dataset_revision": "r1",
                "source_content_hash": "a" * 64,
                "source_task_identities": ["a", "b"],
                "selection_rule": {
                    "kind": "first_n",
                    "params": [["n", "2"]],
                },
            }
        )


def test_repeat_seeds_are_bound_into_repeat_and_plan_identity() -> None:
    assert (
        RepeatId(
            repeat_plan_identity="a" * 64,
            task_identity="task",
            index=0,
        ).identity_hash()
        != RepeatId(
            repeat_plan_identity="a" * 64,
            task_identity="task",
            index=0,
            rng_seed=42,
        ).identity_hash()
    )
    base = RepeatPlan(
        plan_id="p",
        version="1",
        task_identities=("task",),
        repeat_count=2,
    )
    seeded = base.model_copy(
        update={"seeds": (("task#0", 10), ("task#1", 20))}
    )
    assert base.identity_hash() != seeded.identity_hash()


def test_repeat_ids_are_bound_to_their_owning_plan() -> None:
    first_plan = RepeatPlan(
        plan_id="first",
        version="1",
        task_identities=("task",),
        repeat_count=1,
        seeds=(("task#0", 7),),
    )
    second_plan = RepeatPlan(
        plan_id="second",
        version="1",
        task_identities=("task",),
        repeat_count=1,
        seeds=(("task#0", 7),),
    )
    first = first_plan.repeats()[0].repeat_id
    second = second_plan.repeats()[0].repeat_id
    assert first.repeat_plan_identity == first_plan.identity_hash()
    assert second.repeat_plan_identity == second_plan.identity_hash()
    assert first.identity_hash() != second.identity_hash()
    assert RepeatId.model_validate_json(first.model_dump_json()) == first


def test_repeat_plan_rejects_duplicate_seed_slots() -> None:
    with pytest.raises(ValueError, match="seed keys must be unique"):
        RepeatPlan(
            plan_id="p",
            version="1",
            task_identities=("task",),
            repeat_count=1,
            seeds=(("task#0", 10), ("task#0", 20)),
        )


def test_repeat_plan_rejects_unknown_seed_slots() -> None:
    with pytest.raises(ValueError, match="unknown slots"):
        RepeatPlan(
            plan_id="p",
            version="1",
            task_identities=("task",),
            repeat_count=1,
            seeds=(("other#0", 10),),
        )


def test_repeat_id_rejects_negative_index() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RepeatId(
            repeat_plan_identity="a" * 64,
            task_identity="task",
            index=-1,
        )


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: RepeatId(
            repeat_plan_identity="a" * 64,
            task_identity="task",
            index=True,
        ),
        lambda: RepeatId(
            repeat_plan_identity="a" * 64,
            task_identity="task",
            index=0,
            rng_seed=True,
        ),
        lambda: RepeatPlan(
            plan_id="p",
            version="1",
            task_identities=("task",),
            repeat_count=True,
        ),
        lambda: RepeatPlan(
            plan_id="p",
            version="1",
            task_identities=("task",),
            repeat_count=1,
            seeds=(("task#0", True),),
        ),
        lambda: RepeatProvenanceRow("task", True),
        lambda: RepeatProvenanceRow("task", 0, True),
    ],
)
def test_repeat_integer_contracts_reject_bool(constructor) -> None:
    with pytest.raises((TypeError, ValueError), match="integer|valid integer"):
        constructor()


def test_repeat_plan_rejects_noncanonical_seed_order() -> None:
    with pytest.raises(ValueError, match="canonical task-major order"):
        RepeatPlan(
            plan_id="p",
            version="1",
            task_identities=("a", "b"),
            repeat_count=2,
            seeds=(
                ("b#1", 21),
                ("b#0", 20),
                ("a#1", 11),
                ("a#0", 10),
            ),
        )


def test_provenance_materializes_task_major_repeat_slots() -> None:
    plan = repeat_plan_from_provenance(
        (
            RepeatProvenanceRow("a", 0, 4),
            RepeatProvenanceRow("a", 1),
            RepeatProvenanceRow("b", 0, 8),
            RepeatProvenanceRow("b", 1),
        ),
        plan_id="p",
        version="1",
    )
    assert [
        (
            repeat.repeat_id.task_identity,
            repeat.repeat_id.index,
            repeat.repeat_id.rng_seed,
        )
        for repeat in plan.repeats()
    ] == [
        ("a", 0, 4),
        ("a", 1, None),
        ("b", 0, 8),
        ("b", 1, None),
    ]


def test_provenance_rejects_ragged_slots() -> None:
    with pytest.raises(ValueError, match="same number"):
        repeat_plan_from_provenance(
            (
                RepeatProvenanceRow("a", 0),
                RepeatProvenanceRow("a", 1),
                RepeatProvenanceRow("b", 0),
            ),
            plan_id="p",
            version="1",
        )


def test_provenance_rejects_non_contiguous_slots() -> None:
    with pytest.raises(ValueError, match="not contiguous"):
        repeat_plan_from_provenance(
            (
                RepeatProvenanceRow("a", 0),
                RepeatProvenanceRow("a", 2),
            ),
            plan_id="p",
            version="1",
        )


def test_provenance_rejects_duplicate_slots() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        repeat_plan_from_provenance(
            (
                RepeatProvenanceRow("a", 0),
                RepeatProvenanceRow("a", 0),
            ),
            plan_id="p",
            version="1",
        )
