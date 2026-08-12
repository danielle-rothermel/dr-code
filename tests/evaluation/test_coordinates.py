from __future__ import annotations

import pytest
from pydantic import ValidationError

from _builders import (
    candidate,
    dataset,
    evaluation_slot,
    preprocessing_coordinate,
    repeat_plan,
    repeat_plan_coordinate,
    sample_identity,
    task_set,
    task_set_coordinate,
)
from dr_code.evaluation import (
    DatasetCoordinate,
    EvaluationCandidateIdentity,
    EvaluationSampleIdentity,
    EvaluationSlotIdentity,
    RepeatPlan,
    RepeatPlanCoordinate,
    TaskSet,
    TaskSetCoordinate,
)


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
    with pytest.raises(ValidationError, match="preserve population order"):
        task_set(population=("a", "b", "c"), selected=("c", "a"))


def test_repeat_plan_slot_count_sums_the_per_task_repeats() -> None:
    plan = repeat_plan(task_count=3, task_repeats=(4, 1, 2))
    assert plan.slot_count == 7


def test_repeat_plan_slot_count_matches_a_uniform_rectangle() -> None:
    assert repeat_plan(task_count=3, task_repeats=(4,) * 3).slot_count == 12


@pytest.mark.parametrize(
    ("task_position", "repeat_index", "expected"),
    [(0, 0, 0), (0, 2, 2), (1, 0, 3), (2, 2, 8)],
)
def test_slot_index_flattens_task_major(
    task_position: int, repeat_index: int, expected: int
) -> None:
    plan = repeat_plan(task_count=3, task_repeats=(3, 3, 3))
    assert plan.slot_index(task_position, repeat_index) == expected


@pytest.mark.parametrize(
    ("task_position", "repeat_index", "expected"),
    [(0, 0, 0), (0, 2, 2), (1, 0, 3), (2, 0, 4), (2, 1, 5)],
)
def test_slot_index_prefix_sums_ragged_repeats(
    task_position: int, repeat_index: int, expected: int
) -> None:
    plan = repeat_plan(task_count=3, task_repeats=(3, 1, 2))
    assert plan.slot_index(task_position, repeat_index) == expected


def test_slot_index_covers_every_ragged_slot_exactly_once() -> None:
    plan = repeat_plan(task_count=3, task_repeats=(3, 1, 2))
    indices = [
        plan.slot_index(task, repeat)
        for task in range(plan.task_count)
        for repeat in range(plan.repeats_for(task))
    ]
    assert indices == list(range(plan.slot_count))


@pytest.mark.parametrize("task_position", [-1, 3])
def test_slot_index_rejects_a_task_outside_the_plan(
    task_position: int,
) -> None:
    plan = repeat_plan(task_count=3, task_repeats=(2, 2, 2))
    with pytest.raises(ValueError, match="task position"):
        plan.slot_index(task_position, 0)


@pytest.mark.parametrize("repeat_index", [-1, 2])
def test_slot_index_rejects_a_repeat_outside_its_own_task(
    repeat_index: int,
) -> None:
    plan = repeat_plan(task_count=3, task_repeats=(2, 2, 2))
    with pytest.raises(ValueError, match="repeat index"):
        plan.slot_index(0, repeat_index)


def test_slot_index_bounds_each_task_by_its_own_repeat_count() -> None:
    plan = repeat_plan(task_count=2, task_repeats=(3, 1))
    assert plan.slot_index(0, 2) == 2
    with pytest.raises(ValueError, match="repeat index"):
        plan.slot_index(1, 1)


def test_repeat_plan_rejects_a_zero_task_count() -> None:
    with pytest.raises(ValidationError, match="at least one task"):
        repeat_plan(task_count=0, task_repeats=())


def test_repeat_plan_rejects_a_zero_repeat_count() -> None:
    with pytest.raises(ValidationError, match="at least one repeat"):
        repeat_plan(task_count=2, task_repeats=(2, 0))


@pytest.mark.parametrize("task_repeats", [(2,), (2, 2, 2)])
def test_repeat_plan_rejects_repeats_misaligned_with_the_task_count(
    task_repeats: tuple[int, ...],
) -> None:
    with pytest.raises(ValidationError, match="one repeat count per task"):
        repeat_plan(task_count=2, task_repeats=task_repeats)


def test_repeat_plan_seeds_are_optional() -> None:
    assert repeat_plan().seeds is None


def test_repeat_plan_accepts_one_seed_per_slot() -> None:
    plan = repeat_plan(
        task_count=2, task_repeats=(3, 3), seeds=tuple(range(6))
    )
    assert plan.seeds is not None
    assert len(plan.seeds) == plan.slot_count


def test_repeat_plan_seeds_count_follows_ragged_slots() -> None:
    plan = repeat_plan(
        task_count=2, task_repeats=(3, 1), seeds=tuple(range(4))
    )
    assert plan.seeds is not None
    assert len(plan.seeds) == plan.slot_count == 4


@pytest.mark.parametrize("seeds", [(1, 2), (1,) * 5])
def test_repeat_plan_rejects_a_seed_count_that_misses_slots(
    seeds: tuple[int, ...],
) -> None:
    with pytest.raises(ValidationError, match="one seed per slot"):
        repeat_plan(task_count=2, task_repeats=(2, 2), seeds=seeds)


def test_repeat_plan_rejects_empty_seeds_when_slots_exist() -> None:
    with pytest.raises(ValidationError, match="one seed per slot"):
        repeat_plan(task_count=2, task_repeats=(2, 2), seeds=())


def test_evaluation_slot_rejects_a_negative_repeat_index() -> None:
    with pytest.raises(ValidationError, match="repeat_index"):
        evaluation_slot(repeat_index=-1)


def test_candidate_rejects_a_negative_ordinal() -> None:
    with pytest.raises(ValidationError, match="candidate_ordinal"):
        candidate(candidate_ordinal=-1)


def test_candidate_accepts_the_first_ordinal() -> None:
    assert candidate(candidate_ordinal=0).candidate_ordinal == 0


def test_candidate_ordinal_documents_the_post_filter_definition() -> None:
    doc = EvaluationCandidateIdentity.__doc__ or ""
    assert "after" in doc
    assert "materialization" in doc


def test_candidate_nests_sample_identity_and_preprocessing() -> None:
    built = candidate()
    assert built.sample == sample_identity()
    assert built.preprocessing == preprocessing_coordinate()


@pytest.mark.parametrize(
    ("model", "value"),
    [
        (DatasetCoordinate, dataset()),
        (TaskSetCoordinate, task_set_coordinate()),
        (TaskSet, task_set()),
        (RepeatPlanCoordinate, repeat_plan_coordinate()),
        (RepeatPlan, repeat_plan()),
        (RepeatPlan, repeat_plan(seeds=(1, 2, 3, 4))),
        (EvaluationSlotIdentity, evaluation_slot()),
        (EvaluationSampleIdentity, sample_identity()),
        (EvaluationCandidateIdentity, candidate()),
    ],
)
def test_coordinate_round_trips_through_json(model, value) -> None:
    assert model.model_validate_json(value.model_dump_json()) == value
