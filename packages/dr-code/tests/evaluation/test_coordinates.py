from __future__ import annotations

import pytest
from pydantic import ValidationError

from _builders import (
    candidate,
    dataset,
    evaluation_slot,
    preprocessing_coordinate,
    sampling_plan,
    sampling_plan_coordinate,
    sample_id,
    task_set,
    task_set_coordinate,
)
from dr_code.evaluation import (
    DatasetCoordinate,
    EvalCandidateId,
    EvalSampleId,
    EvalSlotId,
    SamplingPlan,
    SamplingPlanCoordinate,
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


def test_sampling_plan_slot_count_sums_the_per_task_samples() -> None:
    plan = sampling_plan(task_count=3, task_num_samples=(4, 1, 2))
    assert plan.slot_count == 7


def test_sampling_plan_slot_count_matches_a_uniform_rectangle() -> None:
    assert (
        sampling_plan(task_count=3, task_num_samples=(4,) * 3).slot_count == 12
    )


@pytest.mark.parametrize(
    ("task_index", "sample_index", "expected"),
    [(0, 0, 0), (0, 2, 2), (1, 0, 3), (2, 2, 8)],
)
def test_slot_index_flattens_task_major(
    task_index: int, sample_index: int, expected: int
) -> None:
    plan = sampling_plan(task_count=3, task_num_samples=(3, 3, 3))
    assert plan.slot_index(task_index, sample_index) == expected


@pytest.mark.parametrize(
    ("task_index", "sample_index", "expected"),
    [(0, 0, 0), (0, 2, 2), (1, 0, 3), (2, 0, 4), (2, 1, 5)],
)
def test_slot_index_prefix_sums_ragged_samples(
    task_index: int, sample_index: int, expected: int
) -> None:
    plan = sampling_plan(task_count=3, task_num_samples=(3, 1, 2))
    assert plan.slot_index(task_index, sample_index) == expected


def test_slot_index_covers_every_ragged_slot_exactly_once() -> None:
    plan = sampling_plan(task_count=3, task_num_samples=(3, 1, 2))
    indices = [
        plan.slot_index(task, sample)
        for task in range(plan.task_count)
        for sample in range(plan.num_samples_for(task))
    ]
    assert indices == list(range(plan.slot_count))


@pytest.mark.parametrize("task_index", [-1, 3])
def test_slot_index_rejects_a_task_outside_the_plan(
    task_index: int,
) -> None:
    plan = sampling_plan(task_count=3, task_num_samples=(2, 2, 2))
    with pytest.raises(ValueError, match="task index"):
        plan.slot_index(task_index, 0)


@pytest.mark.parametrize("sample_index", [-1, 2])
def test_slot_index_rejects_a_sample_outside_its_own_task(
    sample_index: int,
) -> None:
    plan = sampling_plan(task_count=3, task_num_samples=(2, 2, 2))
    with pytest.raises(ValueError, match="sample index"):
        plan.slot_index(0, sample_index)


def test_slot_index_bounds_each_task_by_its_own_sample_count() -> None:
    plan = sampling_plan(task_count=2, task_num_samples=(3, 1))
    assert plan.slot_index(0, 2) == 2
    with pytest.raises(ValueError, match="sample index"):
        plan.slot_index(1, 1)


def test_sampling_plan_rejects_a_zero_task_count() -> None:
    with pytest.raises(ValidationError, match="at least one task"):
        sampling_plan(task_count=0, task_num_samples=())


def test_sampling_plan_rejects_a_zero_sample_count() -> None:
    with pytest.raises(ValidationError, match="at least one sample"):
        sampling_plan(task_count=2, task_num_samples=(2, 0))


@pytest.mark.parametrize("task_num_samples", [(2,), (2, 2, 2)])
def test_sampling_plan_rejects_samples_misaligned_with_the_task_count(
    task_num_samples: tuple[int, ...],
) -> None:
    with pytest.raises(ValidationError, match="one sample count per task"):
        sampling_plan(task_count=2, task_num_samples=task_num_samples)


def test_sampling_plan_seeds_are_optional() -> None:
    assert sampling_plan().seeds is None


def test_sampling_plan_accepts_one_seed_per_slot() -> None:
    plan = sampling_plan(
        task_count=2, task_num_samples=(3, 3), seeds=tuple(range(6))
    )
    assert plan.seeds is not None
    assert len(plan.seeds) == plan.slot_count


def test_sampling_plan_seeds_count_follows_ragged_slots() -> None:
    plan = sampling_plan(
        task_count=2, task_num_samples=(3, 1), seeds=tuple(range(4))
    )
    assert plan.seeds is not None
    assert len(plan.seeds) == plan.slot_count == 4


@pytest.mark.parametrize("seeds", [(1, 2), (1,) * 5])
def test_sampling_plan_rejects_a_seed_count_that_misses_slots(
    seeds: tuple[int, ...],
) -> None:
    with pytest.raises(ValidationError, match="one seed per slot"):
        sampling_plan(task_count=2, task_num_samples=(2, 2), seeds=seeds)


def test_sampling_plan_rejects_empty_seeds_when_slots_exist() -> None:
    with pytest.raises(ValidationError, match="one seed per slot"):
        sampling_plan(task_count=2, task_num_samples=(2, 2), seeds=())


def test_eval_slot_rejects_a_negative_sample_index() -> None:
    with pytest.raises(ValidationError, match="sample_index"):
        evaluation_slot(sample_index=-1)


def test_candidate_rejects_a_negative_ordinal() -> None:
    with pytest.raises(ValidationError, match="candidate_ordinal"):
        candidate(candidate_ordinal=-1)


def test_candidate_accepts_the_first_ordinal() -> None:
    assert candidate(candidate_ordinal=0).candidate_ordinal == 0


def test_candidate_ordinal_documents_the_post_filter_definition() -> None:
    doc = EvalCandidateId.__doc__ or ""
    assert "after" in doc
    assert "materialization" in doc


def test_candidate_nests_sample_id_and_preprocessing() -> None:
    built = candidate()
    assert built.sample == sample_id()
    assert built.preprocessing == preprocessing_coordinate()


@pytest.mark.parametrize(
    ("model", "value"),
    [
        (DatasetCoordinate, dataset()),
        (TaskSetCoordinate, task_set_coordinate()),
        (TaskSet, task_set()),
        (SamplingPlanCoordinate, sampling_plan_coordinate()),
        (SamplingPlan, sampling_plan()),
        (SamplingPlan, sampling_plan(seeds=(1, 2, 3, 4))),
        (EvalSlotId, evaluation_slot()),
        (EvalSampleId, sample_id()),
        (EvalCandidateId, candidate()),
    ],
)
def test_coordinate_round_trips_through_json(model, value) -> None:
    assert model.model_validate_json(value.model_dump_json()) == value
