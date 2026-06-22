"""Read and write AttemptRecord exports."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dr_code.models.attempts import (
    AttemptProvenance,
    AttemptRecord,
    AttemptSource,
)


def write_attempts(records: list[AttemptRecord], path: Path | str) -> Path:
    """Write AttemptRecord rows to Parquet or JSONL."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix == ".parquet":
        _write_parquet(records, out)
    elif suffix == ".jsonl":
        _write_jsonl(records, out)
    else:
        msg = f"Unsupported export format: {suffix} (use .parquet or .jsonl)"
        raise ValueError(msg)
    return out


def read_attempts(path: Path | str) -> list[AttemptRecord]:
    """Read AttemptRecord rows from Parquet or JSONL."""
    in_path = Path(path)
    suffix = in_path.suffix.lower()
    if suffix == ".parquet":
        return _read_parquet(in_path)
    if suffix == ".jsonl":
        return _read_jsonl(in_path)
    msg = f"Unsupported import format: {suffix} (use .parquet or .jsonl)"
    raise ValueError(msg)


def _record_to_row(record: AttemptRecord) -> dict[str, object]:
    provenance = record.provenance
    row: dict[str, object] = {
        "sample_id": record.sample_id,
        "run_id": record.run_id,
        "task_id": record.task_id,
        "entry_point": record.entry_point,
        "decoder_input": record.decoder_input,
        "raw_output": record.raw_output,
        "provenance_source": provenance.source.value,
        "provenance_model": provenance.model,
        "provenance_pool_name": provenance.pool_name,
        "provenance_prompt_template_id": provenance.prompt_template_id,
        "provenance_enc_llm_config_id": provenance.enc_llm_config_id,
        "provenance_dec_llm_config_id": provenance.dec_llm_config_id,
        "provenance_occurrence_count": provenance.occurrence_count,
        "provenance_pool_attempt_id": provenance.pool_attempt_id,
        "provenance_extra_json": json.dumps(provenance.extra, sort_keys=True),
    }
    return row


def _row_to_record(row: dict[str, object]) -> AttemptRecord:
    extra_raw = row.get("provenance_extra_json")
    extra: dict[str, str | int | float | None] = {}
    if isinstance(extra_raw, str) and extra_raw:
        loaded = json.loads(extra_raw)
        if isinstance(loaded, dict):
            extra = loaded
    provenance = AttemptProvenance(
        source=AttemptSource(str(row["provenance_source"])),
        model=_optional_str(row.get("provenance_model")),
        pool_name=_optional_str(row.get("provenance_pool_name")),
        prompt_template_id=_optional_str(
            row.get("provenance_prompt_template_id")
        ),
        enc_llm_config_id=_optional_str(
            row.get("provenance_enc_llm_config_id")
        ),
        dec_llm_config_id=_optional_str(
            row.get("provenance_dec_llm_config_id")
        ),
        occurrence_count=_optional_int(
            row.get("provenance_occurrence_count"),
            default=1,
        ),
        pool_attempt_id=_optional_str(row.get("provenance_pool_attempt_id")),
        extra=extra,
    )
    return AttemptRecord(
        sample_id=str(row["sample_id"]),
        run_id=_optional_str(row.get("run_id")),
        task_id=str(row["task_id"]),
        entry_point=str(row["entry_point"]),
        decoder_input=str(row["decoder_input"]),
        raw_output=str(row["raw_output"]),
        provenance=provenance,
    )


def _optional_int(value: object | None, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return int(str(value))


def _optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _write_parquet(records: list[AttemptRecord], path: Path) -> None:
    rows = [_record_to_row(record) for record in records]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _read_parquet(path: Path) -> list[AttemptRecord]:
    table = pq.read_table(path)
    return [_row_to_record(row) for row in table.to_pylist()]


def _write_jsonl(records: list[AttemptRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
            )
            handle.write("\n")


def _read_jsonl(path: Path) -> list[AttemptRecord]:
    records: list[AttemptRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(AttemptRecord.model_validate(json.loads(line)))
    return records
