"""Import dr-llm HumanEval pool artifacts into AttemptRecord rows."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from dr_code.datasets.humaneval_loader import get_task, task_index
from dr_code.models.attempts import (
    AttemptProvenance,
    AttemptRecord,
    AttemptSource,
    provenance_from_pool_row,
)

_DEDUP_FILENAME_RE = re.compile(r"human_eval-(\d+)-decode-dedup\.jsonl$")


def load_pool_parquet(
    path: Path | str,
    *,
    prefer_snapshot: bool = True,
    limit: int | None = None,
) -> list[AttemptRecord]:
    """Load pool Parquet rows as AttemptRecord instances."""
    table = pq.read_table(str(path))
    rows = table.to_pylist()
    index = task_index(prefer_snapshot=prefer_snapshot)
    records: list[AttemptRecord] = []
    for row in rows:
        task_id = str(row["human_eval_task_id"])
        if task_id not in index:
            msg = f"Unknown task_id in pool parquet: {task_id}"
            raise KeyError(msg)
        entry_point = index[task_id].entry_point
        records.append(
            AttemptRecord.from_pool_row(row, entry_point=entry_point)
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def infer_task_id_from_dedup_path(path: Path | str) -> str | None:
    """Infer HumanEval/N from a dedup JSONL filename."""
    match = _DEDUP_FILENAME_RE.search(Path(path).name)
    if match is None:
        return None
    return f"HumanEval/{match.group(1)}"


def load_pool_dedup_jsonl(
    path: Path | str,
    *,
    task_id: str | None = None,
    decoder_input: str | None = None,
    prefer_snapshot: bool = True,
    limit: int | None = None,
) -> list[AttemptRecord]:
    """Load dedup JSONL rows as AttemptRecord instances."""
    resolved_task_id = task_id or infer_task_id_from_dedup_path(path)
    if resolved_task_id is None:
        msg = (
            "task_id is required for dedup JSONL when filename does not match "
            "human_eval-<n>-decode-dedup.jsonl"
        )
        raise ValueError(msg)
    task = get_task(resolved_task_id, prefer_snapshot=prefer_snapshot)
    resolved_decoder_input = decoder_input or task.prompt
    records: list[AttemptRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            out = str(payload["out"])
            count = int(payload["count"])
            records.append(
                AttemptRecord.from_dedup_row(
                    out=out,
                    count=count,
                    task_id=resolved_task_id,
                    entry_point=task.entry_point,
                    decoder_input=resolved_decoder_input,
                )
            )
            if limit is not None and len(records) >= limit:
                break
    return records


def _lookup_parquet_row(
    parquet_rows: list[dict[str, Any]],
    raw_output: str,
) -> dict[str, Any] | None:
    for row in parquet_rows:
        if str(row["raw_code_output"]) == raw_output:
            return row
    return None


def load_pool_dedup_with_parquet(
    dedup_path: Path | str,
    parquet_path: Path | str,
    *,
    task_id: str | None = None,
    prefer_snapshot: bool = True,
    limit: int | None = None,
) -> list[AttemptRecord]:
    """Load dedup JSONL with optional Parquet provenance join."""
    resolved_task_id = task_id or infer_task_id_from_dedup_path(dedup_path)
    if resolved_task_id is None:
        msg = (
            "task_id is required for dedup JSONL when filename does not match "
            "human_eval-<n>-decode-dedup.jsonl"
        )
        raise ValueError(msg)
    task = get_task(resolved_task_id, prefer_snapshot=prefer_snapshot)
    table = pq.read_table(str(parquet_path))
    parquet_rows = table.to_pylist()
    records: list[AttemptRecord] = []
    with Path(dedup_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            out = str(payload["out"])
            count = int(payload["count"])
            matched = _lookup_parquet_row(parquet_rows, out)
            if matched is not None:
                provenance = provenance_from_pool_row(matched)
                provenance = provenance.model_copy(
                    update={
                        "source": AttemptSource.POOL,
                        "occurrence_count": count,
                    }
                )
                decoder_input = str(matched["decoder_input_description"])
            else:
                provenance = AttemptProvenance(
                    source=AttemptSource.POOL,
                    occurrence_count=count,
                )
                decoder_input = task.prompt
            records.append(
                AttemptRecord.from_dedup_row(
                    out=out,
                    count=count,
                    task_id=resolved_task_id,
                    entry_point=task.entry_point,
                    decoder_input=decoder_input,
                    provenance=provenance,
                )
            )
            if limit is not None and len(records) >= limit:
                break
    return records
