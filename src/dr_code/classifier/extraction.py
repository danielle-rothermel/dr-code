"""Collect classifiable failures from a validated run's artifacts.

Two failure families are extracted:

* Parse/extraction failures: the decoder emitted nonblank text but the pipeline
  produced no final candidate (``final_candidate_count = 0`` with a recorded
  ``failure_code``). The raw decoder-output text is carried for classification.
  This mirrors the waterfall "did not advance" logic in
  :mod:`dr_code.viewer.analytics`.
* Test failures: a candidate compiled and was evaluated but did not pass
  (``outcome != 'passed'``). Only available when candidate-evaluation artifacts
  are registered. The failure message/type is carried as the text.

Items are returned in a stable order (by ``sample_id``) so that caps and
resumption are deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from dr_code.classifier.taxonomy import FailureKind
from dr_code.viewer.domain import RunDescriptor


@dataclass(frozen=True, slots=True)
class FailureItem:
    """One classifiable failure with the text a lane will read.

    ``item_id`` is a stable, run-local identifier unique across a run's failure
    set. ``dataset_id``/``task_id`` locate the benchmark task for the per-task
    rollup; ``task_id`` may be ``None`` when the corpus lacks the column.
    """

    item_id: str
    kind: FailureKind
    sample_id: str
    dataset_id: str | None
    task_id: str | None
    failure_code: str | None
    failed_step: str | None
    text: str


_PARSE_QUERY = """
WITH facts AS (SELECT * FROM read_parquet(?))
SELECT
    c.sample_id,
    {task_expr} AS task_id,
    r.failure_code,
    r.failed_step,
    c.decoder_output
FROM read_parquet(?) AS r
JOIN read_parquet(?) AS c USING (sample_id)
WHERE EXISTS (
        SELECT 1 FROM facts AS sf
        WHERE sf.sample_id = r.sample_id
          AND sf.step_name = 'require_nonblank_text'
          AND cast(json_extract(sf.facts_json, '$.is_nonblank') AS BOOLEAN)
      )
  AND r.final_candidate_count = 0
  AND r.failure_code IS NOT NULL
  AND r.failed_step IS NOT NULL
ORDER BY c.sample_id
"""

_TEST_QUERY = """
SELECT
    em.sample_id,
    em.candidate_id,
    er.task_id,
    er.failure_type,
    er.failure_message,
    er.outcome
FROM read_parquet(?) AS em
JOIN read_parquet(?) AS er USING (evaluation_key)
WHERE er.record_status = 'measured'
  AND er.outcome IS DISTINCT FROM 'passed'
ORDER BY em.sample_id, em.candidate_id
"""


def _dataset_id(task_id: str | None) -> str | None:
    if task_id is None:
        return None
    head, sep, _ = task_id.rpartition("/")
    return head if sep else task_id


def _corpus_has_task_id(descriptor: RunDescriptor) -> bool:
    import pyarrow.parquet as pq

    return "task_id" in pq.ParquetFile(descriptor.corpus_path).schema_arrow.names


def extract_parse_failures(
    connection: duckdb.DuckDBPyConnection,
    descriptor: RunDescriptor,
    *,
    limit: int | None = None,
) -> tuple[list[FailureItem], int]:
    """Return parse/extraction failures and the total available count.

    The returned list is capped to ``limit`` (first N by stable ``sample_id``
    order) while the second element reports the full population size.
    """
    has_task = _corpus_has_task_id(descriptor)
    task_expr = "c.task_id" if has_task else "NULL::VARCHAR"
    rows = connection.execute(
        _PARSE_QUERY.format(task_expr=task_expr),
        [
            str(descriptor.step_facts_path),
            str(descriptor.results_path),
            str(descriptor.corpus_path),
        ],
    ).fetchall()
    total = len(rows)
    selected = rows if limit is None else rows[:limit]
    items = [
        FailureItem(
            item_id=f"parse:{sample_id}",
            kind=FailureKind.PARSE,
            sample_id=sample_id,
            dataset_id=_dataset_id(task_id),
            task_id=task_id,
            failure_code=failure_code,
            failed_step=failed_step,
            text=decoder_output if decoder_output is not None else "",
        )
        for sample_id, task_id, failure_code, failed_step, decoder_output in selected
    ]
    return items, total


def extract_test_failures(
    connection: duckdb.DuckDBPyConnection,
    descriptor: RunDescriptor,
    *,
    limit: int | None = None,
) -> tuple[list[FailureItem], int]:
    """Return compiled-but-failed test failures and the total count.

    Empty (with total ``0``) when the run carries no evaluation artifacts.
    """
    if (
        descriptor.candidate_membership_path is None
        or descriptor.candidate_results_path is None
    ):
        return [], 0
    rows = connection.execute(
        _TEST_QUERY,
        [
            str(descriptor.candidate_membership_path),
            str(descriptor.candidate_results_path),
        ],
    ).fetchall()
    total = len(rows)
    selected = rows if limit is None else rows[:limit]
    items = []
    for (
        sample_id,
        candidate_id,
        task_id,
        failure_type,
        failure_message,
        outcome,
    ) in selected:
        text = _test_failure_text(outcome, failure_type, failure_message)
        items.append(
            FailureItem(
                item_id=f"test:{sample_id}:{candidate_id}",
                kind=FailureKind.TEST,
                sample_id=sample_id,
                dataset_id=_dataset_id(task_id),
                task_id=task_id,
                failure_code=outcome,
                failed_step=None,
                text=text,
            )
        )
    return items, total


def _test_failure_text(
    outcome: str | None,
    failure_type: str | None,
    failure_message: str | None,
) -> str:
    parts = []
    if outcome:
        parts.append(f"outcome: {outcome}")
    if failure_type:
        parts.append(f"failure_type: {failure_type}")
    if failure_message:
        parts.append(f"failure_message: {failure_message}")
    return "\n".join(parts)


__all__ = (
    "FailureItem",
    "extract_parse_failures",
    "extract_test_failures",
)
