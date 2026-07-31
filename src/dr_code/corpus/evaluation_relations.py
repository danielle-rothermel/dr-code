"""Strict reconciliation for schema-5 candidate-evaluation relations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from dr_code.corpus.candidate_evaluation_contract import (
    CandidateEvaluationContractError,
    validate_candidate_result,
)


class EvaluationRelationsError(ValueError):
    """Evaluation relations contradict their immutable coordinates."""


def validate_evaluation_relations(
    *,
    corpus_path: Path,
    candidates_path: Path,
    membership_path: Path,
    results_path: Path,
    coordinates: Mapping[str, object],
) -> None:
    """Reconcile schema-5 rows with corpus, candidates, and the manifest."""

    corpus_columns = pq.ParquetFile(corpus_path).schema_arrow.names
    source_kind = (
        "source_kind" if "source_kind" in corpus_columns else "NULL::VARCHAR"
    )
    connection = duckdb.connect(":memory:")
    try:
        rows = connection.execute(
            f"""
            WITH
            corpus AS (
                SELECT sample_id, task_id, {source_kind} AS source_kind
                FROM read_parquet(?)
            ),
            candidates AS (
                SELECT sample_id, candidate_id, candidate_index, source_sha256
                FROM read_parquet(?)
            ),
            membership AS (SELECT * FROM read_parquet(?)),
            evaluation_results AS (SELECT * FROM read_parquet(?)),
            problems AS (
                SELECT 'duplicate evaluation membership identities' AS label,
                       count(*) - count(DISTINCT (
                           sample_id, candidate_id, candidate_index
                       )) AS n
                FROM membership
                UNION ALL
                SELECT 'candidate/membership identity mismatches', count(*)
                FROM candidates AS c
                FULL OUTER JOIN membership AS m
                    USING (sample_id, candidate_id, candidate_index)
                WHERE c.sample_id IS NULL
                   OR m.sample_id IS NULL
                   OR c.source_sha256 IS DISTINCT FROM m.source_sha256
                UNION ALL
                SELECT 'membership/corpus semantic mismatches', count(*)
                FROM membership AS m
                LEFT JOIN corpus AS c USING (sample_id)
                WHERE c.sample_id IS NULL
                   OR m.task_id IS DISTINCT FROM c.task_id
                   OR m.source_kind IS DISTINCT FROM c.source_kind
                UNION ALL
                SELECT 'membership/manifest coordinate mismatches', count(*)
                FROM membership AS m
                WHERE m.question_identity_hash IS DISTINCT FROM ?
                   OR m.operator_name IS DISTINCT FROM ?
                   OR m.operator_version IS DISTINCT FROM ?
                   OR m.trace_producer_json IS DISTINCT FROM ?
                   OR m.operator_coordinates_json IS DISTINCT FROM ?
                   OR m.metric_extraction_config_identity IS DISTINCT FROM ?
                   OR m.evaluation_procedure_config_identity IS DISTINCT FROM ?
                   OR m.runtime_identity IS DISTINCT FROM ?
                   OR m.runner_identity IS DISTINCT FROM ?
                UNION ALL
                SELECT 'duplicate evaluation result identities',
                       count(*) - count(DISTINCT evaluation_key)
                FROM evaluation_results
                UNION ALL
                SELECT 'membership references to missing evaluation results',
                       count(*)
                FROM membership AS m
                LEFT JOIN evaluation_results AS r USING (evaluation_key)
                WHERE r.evaluation_key IS NULL
                UNION ALL
                SELECT 'unreferenced evaluation results', count(*)
                FROM evaluation_results AS r
                WHERE NOT EXISTS (
                    SELECT 1 FROM membership AS m
                    WHERE m.evaluation_key = r.evaluation_key
                )
                UNION ALL
                SELECT 'evaluation result content fingerprint mismatches',
                       count(*)
                FROM evaluation_results
                WHERE sha256(cleaned_source) IS DISTINCT FROM source_sha256
                UNION ALL
                SELECT 'evaluation result/membership semantic mismatches',
                       count(*)
                FROM membership AS m
                JOIN evaluation_results AS r USING (evaluation_key)
                WHERE r.task_id IS DISTINCT FROM m.task_id
                   OR r.task_identity IS DISTINCT FROM m.task_identity
                   OR r.source_sha256 IS DISTINCT FROM m.source_sha256
                   OR r.question_identity_hash
                      IS DISTINCT FROM m.question_identity_hash
                   OR r.operator_name IS DISTINCT FROM m.operator_name
                   OR r.operator_version IS DISTINCT FROM m.operator_version
                   OR r.trace_producer_json
                      IS DISTINCT FROM m.trace_producer_json
                   OR r.operator_coordinates_json
                      IS DISTINCT FROM m.operator_coordinates_json
                   OR r.metric_extraction_config_identity
                      IS DISTINCT FROM m.metric_extraction_config_identity
                   OR r.evaluation_procedure_config_identity
                      IS DISTINCT FROM m.evaluation_procedure_config_identity
                   OR r.runtime_identity IS DISTINCT FROM m.runtime_identity
                   OR r.runner_identity IS DISTINCT FROM m.runner_identity
                UNION ALL
                SELECT 'evaluation result/manifest coordinate mismatches',
                       count(*)
                FROM evaluation_results AS r
                WHERE r.metrics_profile IS DISTINCT FROM ?
                   OR r.question_identity_hash IS DISTINCT FROM ?
                   OR r.operator_name IS DISTINCT FROM ?
                   OR r.operator_version IS DISTINCT FROM ?
                   OR r.trace_producer_json IS DISTINCT FROM ?
                   OR r.operator_coordinates_json IS DISTINCT FROM ?
                   OR r.metric_extraction_config_identity IS DISTINCT FROM ?
                   OR r.evaluation_procedure_config_identity IS DISTINCT FROM ?
                   OR r.runtime_identity IS DISTINCT FROM ?
                   OR r.runner_identity IS DISTINCT FROM ?
            )
            SELECT label, n FROM problems WHERE n != 0 ORDER BY label
            """,
            [
                str(corpus_path),
                str(candidates_path),
                str(membership_path),
                str(results_path),
                coordinates["question_identity_hash"],
                coordinates["operator_name"],
                coordinates["operator_version"],
                _canonical_json(coordinates["trace_producer"]),
                _canonical_json(coordinates["operator_coordinates"]),
                coordinates["metric_extraction_config_identity"],
                coordinates["evaluation_procedure_config_identity"],
                coordinates["runtime_identity"],
                coordinates["runner_identity"],
                coordinates["metrics_profile"],
                coordinates["question_identity_hash"],
                coordinates["operator_name"],
                coordinates["operator_version"],
                _canonical_json(coordinates["trace_producer"]),
                _canonical_json(coordinates["operator_coordinates"]),
                coordinates["metric_extraction_config_identity"],
                coordinates["evaluation_procedure_config_identity"],
                coordinates["runtime_identity"],
                coordinates["runner_identity"],
            ],
        ).fetchall()
    except duckdb.Error as exc:
        raise EvaluationRelationsError(
            f"candidate evaluation relations could not be reconciled: {exc}"
        ) from exc
    finally:
        connection.close()
    if rows:
        details = ", ".join(f"{label} ({count})" for label, count in rows)
        raise EvaluationRelationsError(
            f"candidate evaluation relational validation failed: {details}"
        )
    for batch in pq.ParquetFile(results_path).iter_batches(batch_size=65_536):
        for row in batch.to_pylist():
            try:
                validate_candidate_result(row)
            except CandidateEvaluationContractError as exc:
                raise EvaluationRelationsError(str(exc)) from exc


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = ("EvaluationRelationsError", "validate_evaluation_relations")
