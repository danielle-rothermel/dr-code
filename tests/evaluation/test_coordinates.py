"""Evaluation coordinate structure and validators.

Covers the task-set population and selection rules, the repeat plan's
uniform task-major slot structure and seed count, the sample coordinate's
repeat bounds, the candidate ordinal bound, and serialization round-trips of
every coordinate model.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from _builders import (
    candidate,
    dataset,
    preprocessing_coordinate,
    repeat_plan,
    repeat_plan_coordinate,
    sample,
    task_set,
    task_set_coordinate,
)
from dr_code.evaluation import (
    CandidateCoordinate,
    DatasetCoordinate,
    RepeatPlan,
    RepeatPlanCoordinate,
    SampleCoordinate,
    TaskSet,
    TaskSetCoordinate,
)

# ===========================================================================
# TaskSet: population and selection rules.
# ===========================================================================


def test_task_set_accepts_an_ordered_subsequence_selection() -> None:
    built = task_set(population=("a", "b", "c", "d"), selected=("a", "c"))
    assert built.selected == ("a", "c")


def test_task_set_accepts_selecting_the_whole_population() -> None:
    built = task_set(population=("a", "b"), selected=("a", "b"))
    assert built.selected == built.population


def test_task_set_rejects_an_empty_population() -> None:
    with pytest.raises(ValidationError, match="population must be non-empty"):
        task_set(population=(), selected=("a",))


def test_task_set_rejects_a_duplicated_population_identity() -> None:
    with pytest.raises(ValidationError, match="population identities"):
        task_set(population=("a", "a"), selected=("a",))


def test_task_set_rejects_an_empty_selection() -> None:
    with pytest.raises(ValidationError, match="selection must be non-empty"):
        task_set(population=("a",), selected=())


def test_task_set_rejects_a_duplicated_selection_identity() -> None:
    with pytest.raises(ValidationError, match="selected identities"):
        task_set(population=("a", "b"), selected=("a", "a"))


def test_task_set_rejects_a_task_outside_the_population() -> None:
    with pytest.raises(ValidationError, match="not in the task set"):
        task_set(population=("a", "b"), selected=("a", "z"))


def test_task_set_rejects_a_selection_that_reorders_the_population() -> None:
    """Selection is a subsequence, not a permutation."""
    with pytest.raises(ValidationError, match="preserve population order"):
        task_set(population=("a", "b", "c"), selected=("c", "a"))


# ===========================================================================
# RepeatPlan: uniform, contiguous, task-major slots and seeds.
# ===========================================================================


def test_repeat_plan_slot_count_is_tasks_times_repeats() -> None:
    assert repeat_plan(task_count=3, repeats=4).slot_count == 12


@pytest.mark.parametrize(
    ("task_position", "repeat_index", "expected"),
    [(0, 0, 0), (0, 2, 2), (1, 0, 3), (2, 2, 8)],
)
def test_slot_index_flattens_task_major(
    task_position: int, repeat_index: int, expected: int
) -> None:
    """All repeats of task 0 precede any repeat of task 1."""
    plan = repeat_plan(task_count=3, repeats=3)
    assert plan.slot_index(task_position, repeat_index) == expected


def test_slot_index_covers_every_slot_exactly_once() -> None:
    plan = repeat_plan(task_count=3, repeats=2)
    indices = [
        plan.slot_index(task, repeat)
        for task in range(plan.task_count)
        for repeat in range(plan.repeats)
    ]
    assert indices == list(range(plan.slot_count))


@pytest.mark.parametrize("task_position", [-1, 3])
def test_slot_index_rejects_a_task_outside_the_plan(
    task_position: int,
) -> None:
    plan = repeat_plan(task_count=3, repeats=2)
    with pytest.raises(ValueError, match="task position"):
        plan.slot_index(task_position, 0)


@pytest.mark.parametrize("repeat_index", [-1, 2])
def test_slot_index_rejects_a_repeat_outside_the_plan(
    repeat_index: int,
) -> None:
    plan = repeat_plan(task_count=3, repeats=2)
    with pytest.raises(ValueError, match="repeat index"):
        plan.slot_index(0, repeat_index)


def test_repeat_plan_rejects_a_zero_task_count() -> None:
    with pytest.raises(ValidationError, match="at least one task"):
        repeat_plan(task_count=0)


def test_repeat_plan_rejects_a_zero_repeat_count() -> None:
    with pytest.raises(ValidationError, match="at least one repeat"):
        repeat_plan(repeats=0)


def test_repeat_plan_seeds_are_optional() -> None:
    assert repeat_plan().seeds is None


def test_repeat_plan_accepts_one_seed_per_slot() -> None:
    plan = repeat_plan(task_count=2, repeats=3, seeds=tuple(range(6)))
    assert plan.seeds is not None
    assert len(plan.seeds) == plan.slot_count


@pytest.mark.parametrize("seeds", [(1, 2), (1,) * 5])
def test_repeat_plan_rejects_a_seed_count_that_misses_slots(
    seeds: tuple[int, ...],
) -> None:
    """A plan seeds every slot or none; partial seeding is rejected."""
    with pytest.raises(ValidationError, match="one seed per slot"):
        repeat_plan(task_count=2, repeats=2, seeds=seeds)


def test_repeat_plan_rejects_empty_seeds_when_slots_exist() -> None:
    with pytest.raises(ValidationError, match="one seed per slot"):
        repeat_plan(task_count=2, repeats=2, seeds=())


# ===========================================================================
# SampleCoordinate: repeat bounds against the plan it names.
# ===========================================================================


def test_sample_rejects_a_negative_repeat_index() -> None:
    with pytest.raises(ValidationError, match="repeat_index"):
        sample(repeat_index=-1)


def test_sample_is_within_a_plan_that_declares_its_slot() -> None:
    plan = repeat_plan(repeats=3)
    assert sample(repeat_index=2).within(plan)


def test_sample_is_outside_a_plan_with_fewer_repeats() -> None:
    plan = repeat_plan(repeats=2)
    assert not sample(repeat_index=2).within(plan)


def test_sample_is_outside_a_plan_it_does_not_name() -> None:
    """Bounds only mean anything against the plan the sample names."""
    other = repeat_plan(
        coordinate=repeat_plan_coordinate(repeat_plan_id="other"), repeats=9
    )
    assert not sample(repeat_index=0).within(other)


# ===========================================================================
# CandidateCoordinate: the ordinal and its documented definition.
# ===========================================================================


def test_candidate_rejects_a_negative_ordinal() -> None:
    with pytest.raises(ValidationError, match="candidate_ordinal"):
        candidate(candidate_ordinal=-1)


def test_candidate_accepts_the_first_ordinal() -> None:
    assert candidate(candidate_ordinal=0).candidate_ordinal == 0


def test_candidate_ordinal_documents_the_post_filter_definition() -> None:
    """The docstring must state the materialized-set definition."""
    doc = CandidateCoordinate.__doc__ or ""
    assert "after" in doc
    assert "deduplication" in doc
    assert "filter" in doc
    assert "materialize_candidate_set" in doc


def test_candidate_nests_the_whole_sample_and_preprocessing() -> None:
    built = candidate()
    assert built.sample == sample()
    assert built.preprocessing == preprocessing_coordinate()


# ===========================================================================
# Serialization round-trips.
# ===========================================================================


@pytest.mark.parametrize(
    ("model", "value"),
    [
        (DatasetCoordinate, dataset()),
        (TaskSetCoordinate, task_set_coordinate()),
        (TaskSet, task_set()),
        (RepeatPlanCoordinate, repeat_plan_coordinate()),
        (RepeatPlan, repeat_plan()),
        (RepeatPlan, repeat_plan(seeds=(1, 2, 3, 4))),
        (SampleCoordinate, sample()),
        (CandidateCoordinate, candidate()),
    ],
)
def test_coordinate_round_trips_through_json(model, value) -> None:
    assert model.model_validate_json(value.model_dump_json()) == value
