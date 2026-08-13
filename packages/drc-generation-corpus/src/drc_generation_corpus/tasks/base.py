from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from drc_generation_corpus.models import DatasetName, TaskRecord


class TaskAdapter(Protocol):
    """Resolve persisted task identities to content-addressed task material."""

    dataset: DatasetName

    def records(self) -> Iterable[TaskRecord]: ...

    def resolve(self, data_sample_id: str) -> TaskRecord | None: ...


__all__ = ["TaskAdapter"]
