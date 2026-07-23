"""Serialization models for the per-example detail JSONL artifact.

Per-example detail (every repeat's label + rationale for every failure item)
lives in a deterministic JSONL file next to the viewer database, not in new
DB columns. These Pydantic models are the read/write boundary for that file.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RepeatRecord(BaseModel):
    """One repeat's outcome as persisted to the detail JSONL."""

    model_config = ConfigDict(extra="forbid")

    index: int
    label: str | None
    rationale: str | None
    failure_reason: str | None = None


class ItemRecord(BaseModel):
    """One failure item's full classification record."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    kind: str
    sample_id: str
    dataset_id: str | None
    task_id: str | None
    failure_code: str | None
    failed_step: str | None
    taxonomy_version: str
    model: str
    lane: str
    repeats: int
    majority_label: str | None
    agreement: float | None
    tie: bool
    successful_repeats: int
    failed_repeats: int
    label_counts: dict[str, int]
    repeat_records: list[RepeatRecord]


__all__ = ("ItemRecord", "RepeatRecord")
