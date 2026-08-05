"""Registry-free coordinates naming what an evaluation measured.

Every model here is a *coordinate*: a complete, self-describing address for
one point in an evaluation, carrying enough structure that a persisted
artifact can be interpreted without consulting a registry, a dataset, or the
plan that produced it. Nesting is deliberate — a ``CandidateCoordinate``
contains its sample, which contains its task set and repeat plan — so no
consumer has to reassemble identity from separately-stored parts.

The two *plan* models here, ``TaskSet`` and ``RepeatPlan``, are the resolved
values their coordinates project. A coordinate names a plan; the plan itself
carries the population and slot structure that makes the naming checkable.
"""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from dr_code.base import FrozenModel
from dr_code.trace import PreprocessingDefinitionCoordinate


class DatasetCoordinate(FrozenModel):
    """The versioned dataset a task population is drawn from."""

    dataset_id: str
    version: str


class TaskSetCoordinate(FrozenModel):
    """The versioned identity of one task set."""

    task_set_id: str
    version: str
    dataset: DatasetCoordinate


class TaskSet(FrozenModel):
    """An ordered selection of tasks drawn from a dataset population.

    ``population`` is the ordered source the selection was made from — the
    dataset's task identities as the selection saw them, recorded so a task
    set stays interpretable after the dataset moves on. ``selected`` is the
    subset actually evaluated.

    Selection preserves population order: ``selected`` is a subsequence of
    ``population``, not an arbitrary permutation of it. Order carries no
    semantics of its own beyond reproducibility, so pinning it to the
    population's order removes a degree of freedom that could otherwise let
    two task sets with identical content compare unequal. A deliberately
    reordered evaluation is a different population, not a resorted
    selection.
    """

    coordinate: TaskSetCoordinate
    population: tuple[str, ...]
    selected: tuple[str, ...]

    @model_validator(mode="after")
    def validate_population(self) -> Self:
        if not self.population:
            raise ValueError("a task set population must be non-empty")
        if len(set(self.population)) != len(self.population):
            raise ValueError("task set population identities must be unique")
        return self

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if not self.selected:
            raise ValueError("a task set selection must be non-empty")
        if len(set(self.selected)) != len(self.selected):
            raise ValueError("task set selected identities must be unique")
        positions: list[int] = []
        index = {task: at for at, task in enumerate(self.population)}
        for task in self.selected:
            if task not in index:
                raise ValueError(
                    f"selected task {task!r} is not in the task set population"
                )
            positions.append(index[task])
        if positions != sorted(positions):
            raise ValueError(
                "selected tasks must preserve population order; "
                f"{self.selected!r} is not a subsequence of the population"
            )
        return self


class RepeatPlanCoordinate(FrozenModel):
    """The versioned identity of one repeat plan."""

    repeat_plan_id: str
    version: str


class RepeatPlan(FrozenModel):
    """A uniform, contiguous, task-major set of repeat slots.

    *Uniform*: every task gets exactly ``repeats`` slots — the plan has no
    per-task counts. *Contiguous*: a task's repeat indices are
    ``0 .. repeats - 1`` with no gaps and no offset. *Task-major*: flattened
    over a task set's ``selected`` order, the slots run as all repeats of
    the first selected task, then all repeats of the second, and so on. Slot
    ``i`` of the flattened sequence therefore addresses selected task
    ``i // repeats`` at repeat index ``i % repeats``.

    ``seeds`` is optional. When present it is the flattened, task-major
    sequence of per-slot seeds, so its length must be exactly
    ``task_count * repeats`` — a plan either seeds every slot or seeds none.
    ``task_count`` is carried on the plan so seed structure is checkable
    without the task set in hand; a ``SampleCoordinate`` is what ties a plan
    back to a specific selection.
    """

    coordinate: RepeatPlanCoordinate
    task_count: int
    repeats: int
    seeds: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def validate_slot_structure(self) -> Self:
        if self.task_count < 1:
            raise ValueError("a repeat plan must cover at least one task")
        if self.repeats < 1:
            raise ValueError("a repeat plan must have at least one repeat")
        return self

    @model_validator(mode="after")
    def validate_seed_count(self) -> Self:
        if self.seeds is None:
            return self
        if len(self.seeds) != self.slot_count:
            raise ValueError(
                f"a seeded repeat plan needs one seed per slot: expected "
                f"{self.slot_count}, got {len(self.seeds)}"
            )
        return self

    @property
    def slot_count(self) -> int:
        """How many repeat slots the plan describes in total."""

        return self.task_count * self.repeats

    def slot_index(self, task_position: int, repeat_index: int) -> int:
        """Flatten a (task position, repeat index) pair task-major."""

        if not 0 <= task_position < self.task_count:
            raise ValueError(
                f"task position {task_position} is outside the plan's "
                f"{self.task_count} tasks"
            )
        if not 0 <= repeat_index < self.repeats:
            raise ValueError(
                f"repeat index {repeat_index} is outside the plan's "
                f"{self.repeats} repeats"
            )
        return task_position * self.repeats + repeat_index


class SampleCoordinate(FrozenModel):
    """One evaluated task at one repeat index under one plan pair.

    A sample is the unit a producer emits one trace for. It nests the
    complete task-set and repeat-plan coordinates rather than their ids, so
    a sample read back in isolation still names the dataset it came from and
    the slot structure its repeat index indexes.
    """

    task_set: TaskSetCoordinate
    repeat_plan: RepeatPlanCoordinate
    task: str
    repeat_index: int

    @model_validator(mode="after")
    def validate_repeat_index(self) -> Self:
        if self.repeat_index < 0:
            raise ValueError("repeat_index must be non-negative")
        return self

    def within(self, plan: RepeatPlan) -> bool:
        """Whether this sample's repeat index is a slot ``plan`` declares.

        The coordinate can only bound its own index from below; the upper
        bound lives on the plan, which a coordinate names but does not
        carry. Callers holding the resolved plan check against it here.
        """

        return (
            self.repeat_plan == plan.coordinate
            and 0 <= self.repeat_index < plan.repeats
        )


class CandidateCoordinate(FrozenModel):
    """One preprocessed candidate produced for one sample.

    ``candidate_ordinal`` is the candidate's zero-based index into the
    materialized candidate set — the set as it stands *after* exact-source
    deduplication and *after* every filter. It does not index the extracted
    set, any intermediate set, or positions before a duplicate was merged
    away. This is the definition ``MaterializeCandidateSet`` establishes
    (``dr_code.preprocessing.steps.materialize_candidate_set``); an ordinal
    is only meaningful against the set that step materialized under the
    nested ``preprocessing`` definition.
    """

    sample: SampleCoordinate
    preprocessing: PreprocessingDefinitionCoordinate
    candidate_ordinal: int

    @model_validator(mode="after")
    def validate_candidate_ordinal(self) -> Self:
        if self.candidate_ordinal < 0:
            raise ValueError("candidate_ordinal must be non-negative")
        return self


__all__ = [
    "CandidateCoordinate",
    "DatasetCoordinate",
    "RepeatPlan",
    "RepeatPlanCoordinate",
    "SampleCoordinate",
    "TaskSet",
    "TaskSetCoordinate",
]
