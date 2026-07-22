"""Task identity, Task Set manifest, and Repeat artifacts.

The Task *role* is implemented by dataset-specific types (here,
``HumanEvalTask``) with **no generic Task superclass**. This module adds:

- a stable :func:`humaneval_task_identity` (Identity Document over the
  dataset-defining fields);
- a versioned :class:`TaskSet` manifest carrying an ordered collection of
  task identities *or* a deterministic selection rule, plus the dataset
  revision;
- :class:`RepeatPlan`, :class:`Repeat`, and :class:`RepeatId` artifacts:
  deliberate independent observations, where an optional RNG seed is slot
  data, never a retry attempt number.
"""

from __future__ import annotations

from typing import Self

from pydantic import field_validator, model_validator

from dr_code.eval.identity import (
    SCHEMA_HUMANEVAL_TASK,
    SCHEMA_REPEAT_ID,
    SCHEMA_REPEAT_PLAN,
    SCHEMA_TASK_SET,
    identity_hash_for,
)
from dr_code.humaneval.task import HumanEvalTask
from dr_code.models import FrozenModel

# The dataset-defining fields of a HumanEvalTask. ``notes`` and the parsed
# derived views are annotation/cache, not identity.
_HUMANEVAL_IDENTITY_FIELDS = (
    "task_id",
    "prompt",
    "canonical_solution",
    "entry_point",
    "test",
)


def humaneval_task_identity_payload(task: HumanEvalTask) -> dict[str, str]:
    """The complete identity payload for one HumanEval task."""

    return {field: getattr(task, field) for field in _HUMANEVAL_IDENTITY_FIELDS}


def humaneval_task_identity(task: HumanEvalTask) -> str:
    """Stable Identity Hash of one ``HumanEvalTask``."""

    return identity_hash_for(
        schema=SCHEMA_HUMANEVAL_TASK,
        payload=humaneval_task_identity_payload(task),
    )


class SelectionRule(FrozenModel):
    """A deterministic selection rule over a dataset revision.

    ``kind`` names the rule family; ``params`` carries its ordered
    parameters. A rule is an alternative to an explicit identity list; a
    Task Set carries exactly one of the two.
    """

    kind: str
    params: tuple[tuple[str, str], ...] = ()


class TaskSet(FrozenModel):
    """Versioned manifest of dataset-specific task identities.

    Exactly one of ``task_identities`` (ordered explicit selection) or
    ``selection_rule`` (deterministic rule) is set. The dataset revision
    and the ordering/selection semantics are all identity-bearing: two
    manifests with the same members in a different order have different
    identities.
    """

    manifest_id: str
    version: str
    dataset_revision: str
    task_identities: tuple[str, ...] = ()
    selection_rule: SelectionRule | None = None

    @model_validator(mode="after")
    def _exactly_one_selector(self) -> Self:
        has_list = len(self.task_identities) > 0
        has_rule = self.selection_rule is not None
        if has_list == has_rule:
            raise ValueError(
                "TaskSet carries exactly one of task_identities or "
                "selection_rule"
            )
        if has_list and len(set(self.task_identities)) != len(
            self.task_identities
        ):
            raise ValueError("task_identities must be unique")
        return self

    def identity_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "manifest_id": self.manifest_id,
            "version": self.version,
            "dataset_revision": self.dataset_revision,
        }
        if self.selection_rule is not None:
            payload["selection_rule"] = {
                "kind": self.selection_rule.kind,
                "params": [list(pair) for pair in self.selection_rule.params],
            }
        else:
            # Ordering is identity-bearing: preserved as a list, not a set.
            payload["task_identities"] = list(self.task_identities)
        return payload

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_TASK_SET,
            payload=self.identity_payload(),
        )


class RepeatId(FrozenModel):
    """Stable identity of one Repeat slot in a Repeat Plan.

    Identified by ``(task_identity, index)``. An optional ``rng_seed`` is
    *slot data* associated with the observation, NOT part of the slot
    identity and never a retry attempt number.
    """

    task_identity: str
    index: int
    rng_seed: int | None = None

    @field_validator("index")
    @classmethod
    def _nonnegative_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("repeat index must be non-negative")
        return value

    def identity_payload(self) -> dict[str, object]:
        # rng_seed is deliberately excluded: it is slot data, not identity.
        return {"task_identity": self.task_identity, "index": self.index}

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_REPEAT_ID,
            payload=self.identity_payload(),
        )


class Repeat(FrozenModel):
    """One deliberate independent observation of a task.

    Distinct from every transport invocation, semantic retry, provider
    attempt, and platform attempt: a Repeat is a planned observation slot.
    """

    repeat_id: RepeatId


class RepeatPlan(FrozenModel):
    """Deterministic ordered plan of deliberate observations per task.

    For each selected task identity the plan enumerates ``repeat_count``
    ordered Repeat slots. Identity covers the task identities, the count,
    and any per-slot seeds.
    """

    plan_id: str
    version: str
    task_identities: tuple[str, ...]
    repeat_count: int
    seeds: tuple[tuple[str, int], ...] = ()

    @field_validator("repeat_count")
    @classmethod
    def _positive_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("repeat_count must be at least 1")
        return value

    @model_validator(mode="after")
    def _unique_tasks(self) -> Self:
        if len(set(self.task_identities)) != len(self.task_identities):
            raise ValueError("task_identities must be unique")
        return self

    def _seed_for(self, task_identity: str, index: int) -> int | None:
        key = f"{task_identity}#{index}"
        for seed_key, seed in self.seeds:
            if seed_key == key:
                return seed
        return None

    def repeats(self) -> tuple[Repeat, ...]:
        """Expand the plan into ordered Repeat slots (task-major)."""

        expanded: list[Repeat] = []
        for task_identity in self.task_identities:
            for index in range(self.repeat_count):
                expanded.append(
                    Repeat(
                        repeat_id=RepeatId(
                            task_identity=task_identity,
                            index=index,
                            rng_seed=self._seed_for(task_identity, index),
                        )
                    )
                )
        return tuple(expanded)

    def identity_payload(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "version": self.version,
            "task_identities": list(self.task_identities),
            "repeat_count": self.repeat_count,
            "seeds": [[key, seed] for key, seed in self.seeds],
        }

    def identity_hash(self) -> str:
        return identity_hash_for(
            schema=SCHEMA_REPEAT_PLAN,
            payload=self.identity_payload(),
        )


__all__ = [
    "Repeat",
    "RepeatId",
    "RepeatPlan",
    "SelectionRule",
    "TaskSet",
    "humaneval_task_identity",
    "humaneval_task_identity_payload",
]
