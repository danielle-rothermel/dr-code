"""Repair base class and `RepairResult` model."""

from __future__ import annotations

from typing import ClassVar

from code_eval.models.base import FrozenModel
from code_eval.names import RepairName


class RepairResult(FrozenModel):
    """The output of applying one repair to a source string.

    `applied_tags` is an ordered tuple of fine-grained operation tags
    (e.g. `"import_recovery:syntactic"`, `"import_recovery:inferred"`).
    The validator surfaces these in `Candidate.repairs_applied` so the
    attribution metric can match against `expected_recovery_steps`.

    `changed` is True iff the repair actually modified the source.
    """

    source: str
    applied_tags: tuple[str, ...]
    changed: bool


class Repair:
    """Base class for repairs.

    Repairs are pure functions of source -> RepairResult. They never raise
    on bad input — if the repair doesn't apply, return `changed=False`.
    """

    NAME: ClassVar[RepairName]

    def apply(self, source: str) -> RepairResult:
        raise NotImplementedError
