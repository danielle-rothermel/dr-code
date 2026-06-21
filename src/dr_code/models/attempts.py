"""AttemptRecord and provenance models for stage 1."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator

from dr_code.models.base import FrozenModel
from dr_code.models.humaneval import HumanEvalPlusTask

_KNOWN_PROVENANCE_KEYS = frozenset(
    {
        "source",
        "model",
        "pool_name",
        "prompt_template_id",
        "enc_llm_config_id",
        "dec_llm_config_id",
        "occurrence_count",
        "pool_attempt_id",
        "extra",
    }
)

_POOL_PROVENANCE_MAP = {
    "model": "model",
    "pool_name": "pool_name",
    "prompt_template_id": "prompt_template_id",
    "enc_llm_config_id": "enc_llm_config_id",
    "dec_llm_config_id": "dec_llm_config_id",
    "attempt_id": "pool_attempt_id",
}


class AttemptSource(StrEnum):
    """Origin of a decoder attempt row."""

    POOL = "pool"
    FRESH_STUB = "fresh_stub"


def compute_sample_id(task_id: str, raw_output: str) -> str:
    """Deterministic sample id shared by pool and fresh sources."""
    payload = f"{task_id}\0{raw_output}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


class AttemptProvenance(FrozenModel):
    """Provenance metadata for an attempt row."""

    source: AttemptSource
    model: str | None = None
    pool_name: str | None = None
    prompt_template_id: str | None = None
    enc_llm_config_id: str | None = None
    dec_llm_config_id: str | None = None
    occurrence_count: int = 1
    pool_attempt_id: str | None = None
    extra: dict[str, str | int | float | None] = Field(default_factory=dict)


class AttemptRecord(FrozenModel):
    """Unified decoder attempt row for stages 2–4."""

    sample_id: str
    run_id: str | None
    task_id: str
    entry_point: str
    decoder_input: str
    raw_output: str
    provenance: AttemptProvenance

    @field_validator("decoder_input", "raw_output")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        if not value:
            msg = "decoder_input and raw_output must be non-empty"
            raise ValueError(msg)
        return value

    @classmethod
    def from_pool_row(
        cls,
        row: dict[str, Any],
        *,
        entry_point: str,
    ) -> AttemptRecord:
        """Build an AttemptRecord from a pool Parquet row dict."""
        task_id = str(row["human_eval_task_id"])
        decoder_input = str(row["decoder_input_description"])
        raw_output = str(row["raw_code_output"])
        provenance = provenance_from_pool_row(row)
        return cls(
            sample_id=compute_sample_id(task_id, raw_output),
            run_id=None,
            task_id=task_id,
            entry_point=entry_point,
            decoder_input=decoder_input,
            raw_output=raw_output,
            provenance=provenance,
        )

    @classmethod
    def from_dedup_row(
        cls,
        *,
        out: str,
        count: int,
        task_id: str,
        entry_point: str,
        decoder_input: str,
        provenance: AttemptProvenance | None = None,
    ) -> AttemptRecord:
        """Build an AttemptRecord from a dedup JSONL row."""
        prov = provenance or AttemptProvenance(source=AttemptSource.POOL)
        return cls(
            sample_id=compute_sample_id(task_id, out),
            run_id=None,
            task_id=task_id,
            entry_point=entry_point,
            decoder_input=decoder_input,
            raw_output=out,
            provenance=prov.model_copy(
                update={
                    "source": AttemptSource.POOL,
                    "occurrence_count": count,
                }
            ),
        )

    @classmethod
    def stub_for_fresh(
        cls,
        task: HumanEvalPlusTask,
        *,
        decoder_input: str,
        raw_output: str,
        run_id: str,
        model: str,
    ) -> AttemptRecord:
        """Build a fresh_stub AttemptRecord for stage 1b generation."""
        return cls(
            sample_id=compute_sample_id(task.task_id, raw_output),
            run_id=run_id,
            task_id=task.task_id,
            entry_point=task.entry_point,
            decoder_input=decoder_input,
            raw_output=raw_output,
            provenance=AttemptProvenance(
                source=AttemptSource.FRESH_STUB,
                model=model,
            ),
        )


def provenance_from_pool_row(row: dict[str, Any]) -> AttemptProvenance:
    """Build AttemptProvenance from a pool Parquet row dict."""
    return _provenance_from_pool_row(row)


def _provenance_from_pool_row(row: dict[str, Any]) -> AttemptProvenance:
    known: dict[str, Any] = {"source": AttemptSource.POOL}
    extra: dict[str, str | int | float | None] = {}
    for col, field_name in _POOL_PROVENANCE_MAP.items():
        if col not in row:
            continue
        value = row[col]
        if value is None:
            continue
        if field_name == "pool_attempt_id":
            known["pool_attempt_id"] = str(value)
        elif field_name in _KNOWN_PROVENANCE_KEYS:
            known[field_name] = value
        else:
            extra[col] = _coerce_extra_value(value)
    for col, value in row.items():
        if col in {
            "human_eval_task_id",
            "decoder_input_description",
            "raw_code_output",
            *_POOL_PROVENANCE_MAP,
        }:
            continue
        if value is None:
            continue
        extra[col] = _coerce_extra_value(value)
    known["extra"] = extra
    return AttemptProvenance.model_validate(known)


def _coerce_extra_value(value: Any) -> str | int | float | None:
    if isinstance(value, (str, int, float)):
        return value
    return str(value)
