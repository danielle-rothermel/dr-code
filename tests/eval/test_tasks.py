"""Task identity, Task Set manifest, and Repeat artifact tests."""

from __future__ import annotations

import pytest

from dr_code.eval.tasks import (
    RepeatId,
    RepeatPlan,
    RepeatProvenanceRow,
    SelectionRule,
    TaskSet,
    humaneval_task_identity,
    repeat_plan_from_provenance,
)
from dr_code.humaneval.task import HumanEvalTask


def _task(task_id: str = "HumanEval/0") -> HumanEvalTask:
    return HumanEvalTask(
        task_id=task_id,
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


def test_task_identity_is_stable_full_sha256() -> None:
    identity = humaneval_task_identity(_task())
    assert len(identity) == 64
    assert identity == humaneval_task_identity(_task())


def test_task_identity_ignores_annotation_fields() -> None:
    base = _task()
    annotated = _task().model_copy(update={"notes": ["a benchmark note"]})
    assert humaneval_task_identity(base) == humaneval_task_identity(annotated)


def test_task_identity_changes_with_dataset_content() -> None:
    a = humaneval_task_identity(_task("HumanEval/0"))
    b = humaneval_task_identity(_task("HumanEval/1"))
    assert a != b


def test_task_set_requires_exactly_one_selector() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        TaskSet(
            manifest_id="m",
            version="1",
            dataset_revision="r1",
            task_identities=("h1",),
            selection_rule=SelectionRule(kind="all"),
        )
    with pytest.raises(ValueError, match="exactly one"):
        TaskSet(manifest_id="m", version="1", dataset_revision="r1")


def test_task_set_ordering_is_identity_bearing() -> None:
    forward = TaskSet(
        manifest_id="m",
        version="1",
        dataset_revision="r1",
        task_identities=("h1", "h2"),
    )
    reversed_ = TaskSet(
        manifest_id="m",
        version="1",
        dataset_revision="r1",
        task_identities=("h2", "h1"),
    )
    assert forward.identity_hash() != reversed_.identity_hash()


def test_task_set_selection_rule_manifest_hashes() -> None:
    manifest = TaskSet(
        manifest_id="m",
        version="1",
        dataset_revision="r1",
        selection_rule=SelectionRule(kind="first_n", params=(("n", "10"),)),
    )
    assert len(manifest.identity_hash()) == 64


def test_repeat_id_seed_is_slot_data_not_identity() -> None:
    without = RepeatId(task_identity="h1", index=0)
    with_seed = RepeatId(task_identity="h1", index=0, rng_seed=42)
    # An optional RNG seed is slot data, excluded from the slot identity.
    assert without.identity_hash() == with_seed.identity_hash()


def test_repeat_id_index_is_identity_bearing() -> None:
    first = RepeatId(task_identity="h1", index=0)
    second = RepeatId(task_identity="h1", index=1)
    assert first.identity_hash() != second.identity_hash()


def test_repeat_plan_expands_deterministic_ordered_slots() -> None:
    plan = RepeatPlan(
        plan_id="p",
        version="1",
        task_identities=("h1", "h2"),
        repeat_count=3,
        seeds=(("h1#0", 7),),
    )
    repeats = plan.repeats()
    assert len(repeats) == 6  # 2 tasks x 3 repeats
    assert [r.repeat_id.index for r in repeats] == [0, 1, 2, 0, 1, 2]
    assert repeats[0].repeat_id.rng_seed == 7
    assert repeats[1].repeat_id.rng_seed is None


def test_repeat_plan_seeds_are_slot_data_not_identity() -> None:
    without = RepeatPlan(
        plan_id="p",
        version="1",
        task_identities=("h1",),
        repeat_count=1,
    )
    with_seed = RepeatPlan(
        plan_id="p",
        version="1",
        task_identities=("h1",),
        repeat_count=1,
        seeds=(("h1#0", 42),),
    )
    # Per-slot RNG seeds are slot data, excluded from plan identity.
    assert without.identity_hash() == with_seed.identity_hash()


def test_repeat_plan_rejects_zero_repeats() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RepeatPlan(
            plan_id="p",
            version="1",
            task_identities=("h1",),
            repeat_count=0,
        )


def test_repeat_plan_from_provenance_materializes_faithfully() -> None:
    # Generation-corpus provenance rows reconstruct the plan: task order is
    # first-seen, repeat_count is the per-task slot count, seeds are carried.
    rows = (
        RepeatProvenanceRow(task_identity="h1", repeat_index=0, seed=7),
        RepeatProvenanceRow(task_identity="h1", repeat_index=1),
        RepeatProvenanceRow(task_identity="h2", repeat_index=0, seed=99),
        RepeatProvenanceRow(task_identity="h2", repeat_index=1),
    )
    plan = repeat_plan_from_provenance(rows, plan_id="p", version="1")
    assert plan.task_identities == ("h1", "h2")
    assert plan.repeat_count == 2
    repeats = plan.repeats()
    assert [r.repeat_id.index for r in repeats] == [0, 1, 0, 1]
    assert repeats[0].repeat_id.rng_seed == 7
    assert repeats[1].repeat_id.rng_seed is None
    assert repeats[2].repeat_id.rng_seed == 99


def test_materialized_plan_identity_is_invariant_under_seeds() -> None:
    # Golden: identical corpus provenance except for differing seeds must
    # produce the SAME RepeatPlan identity (seeds are excluded from identity,
    # consistent with commit 9a597c6).
    base = (
        RepeatProvenanceRow(task_identity="h1", repeat_index=0, seed=1),
        RepeatProvenanceRow(task_identity="h1", repeat_index=1, seed=2),
    )
    reseeded = (
        RepeatProvenanceRow(task_identity="h1", repeat_index=0, seed=555),
        RepeatProvenanceRow(task_identity="h1", repeat_index=1, seed=999),
    )
    unseeded = (
        RepeatProvenanceRow(task_identity="h1", repeat_index=0),
        RepeatProvenanceRow(task_identity="h1", repeat_index=1),
    )
    id_base = repeat_plan_from_provenance(
        base, plan_id="p", version="1"
    ).identity_hash()
    id_reseeded = repeat_plan_from_provenance(
        reseeded, plan_id="p", version="1"
    ).identity_hash()
    id_unseeded = repeat_plan_from_provenance(
        unseeded, plan_id="p", version="1"
    ).identity_hash()
    assert id_base == id_reseeded == id_unseeded


def test_repeat_plan_from_provenance_rejects_ragged_slots() -> None:
    rows = (
        RepeatProvenanceRow(task_identity="h1", repeat_index=0),
        RepeatProvenanceRow(task_identity="h1", repeat_index=1),
        RepeatProvenanceRow(task_identity="h2", repeat_index=0),
    )
    with pytest.raises(ValueError, match="same number of repeat slots"):
        repeat_plan_from_provenance(rows, plan_id="p", version="1")


def test_repeat_plan_from_provenance_rejects_noncontiguous_indices() -> None:
    rows = (
        RepeatProvenanceRow(task_identity="h1", repeat_index=0),
        RepeatProvenanceRow(task_identity="h1", repeat_index=2),
    )
    with pytest.raises(ValueError, match="not contiguous"):
        repeat_plan_from_provenance(rows, plan_id="p", version="1")


def test_repeat_plan_from_provenance_rejects_duplicate_slot() -> None:
    rows = (
        RepeatProvenanceRow(task_identity="h1", repeat_index=0),
        RepeatProvenanceRow(task_identity="h1", repeat_index=0),
    )
    with pytest.raises(ValueError, match="duplicate"):
        repeat_plan_from_provenance(rows, plan_id="p", version="1")
