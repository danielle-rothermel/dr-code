"""Seed AttemptRecord rows from pool dump artifacts."""

from __future__ import annotations

from pathlib import Path

from dr_code.datasets.pool_loader import load_pool_dedup_with_parquet
from dr_code.models.attempts import AttemptRecord
from dr_code.pipeline.jobs import stamp_run_id

DEFAULT_DUMP_DIR = Path(
    "/Users/daniellerothermel/drotherm/data/code-comp/"
    "dr-llm-humaneval-pool-dumps/20260621_manual"
)
DEFAULT_PROOF_INDICES = (0, 1, 2, 3, 4)


def load_proof_attempts(
    dump_dir: Path | str,
    task_indices: list[int] | tuple[int, ...],
    *,
    run_id: str,
    limit_per_task: int | None = None,
) -> list[AttemptRecord]:
    """Load dedup+parquet pool rows for the given HumanEval task indices."""
    root = Path(dump_dir)
    per_elem = root / "per_elem"
    records: list[AttemptRecord] = []
    for index in task_indices:
        dedup_path = per_elem / f"human_eval-{index}-decode-dedup.jsonl"
        parquet_path = per_elem / f"human_eval-{index}-decode.parquet"
        if not dedup_path.is_file():
            msg = f"Missing dedup file: {dedup_path}"
            raise FileNotFoundError(msg)
        if not parquet_path.is_file():
            msg = f"Missing parquet file: {parquet_path}"
            raise FileNotFoundError(msg)
        task_records = load_pool_dedup_with_parquet(
            dedup_path,
            parquet_path,
            limit=limit_per_task,
        )
        records.extend(task_records)
    return stamp_run_id(records, run_id)
