"""Deterministic compact summaries of immutable preprocessing runs."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import pyarrow as pa
import pyarrow.parquet as pq

from dr_code.corpus.atomic_directory import staged_output_directory
from dr_code.corpus.candidate_evaluation_contract import (
    CANDIDATE_EVALUATION_SCHEMA_VERSION,
)
from dr_code.corpus.preprocessing_artifacts import PROJECTED_ARTIFACT_SCHEMAS
from dr_code.corpus.run_descriptor import (
    RunDescriptor,
    RunValidationError,
    admitted_run_descriptor,
    file_sha256,
    normalize_origins,
    thaw_json,
)

ANALYSIS_SCHEMA_VERSION: Final = 3
SUMMARY_FILENAME: Final = "summary.json"
MANIFEST_FILENAME: Final = "analysis_manifest.json"
_ROW_GROUP_SIZE: Final = 65_536

OUTCOMES_SCHEMA: Final = pa.schema(
    [
        pa.field("outcome", pa.string(), nullable=False),
        pa.field("sample_count", pa.int64(), nullable=False),
        pa.field("sample_rate", pa.float64(), nullable=False),
    ]
)
FAILURES_SCHEMA: Final = pa.schema(
    [
        pa.field("failure_code", pa.string(), nullable=False),
        pa.field("failed_step", pa.string(), nullable=False),
        pa.field("cause", pa.string()),
        pa.field("sample_count", pa.int64(), nullable=False),
    ]
)
MULTIPLICITY_SCHEMA: Final = pa.schema(
    [
        pa.field("candidate_count", pa.int64(), nullable=False),
        pa.field("sample_count", pa.int64(), nullable=False),
        pa.field("sample_rate", pa.float64(), nullable=False),
    ]
)
ORIGINS_SCHEMA: Final = pa.schema(
    [
        pa.field("path_json", pa.string(), nullable=False),
        pa.field("candidate_count", pa.int64(), nullable=False),
    ]
)
EVALUATION_OUTCOMES_SCHEMA: Final = pa.schema(
    [
        pa.field("outcome", pa.string(), nullable=False),
        pa.field("candidate_count", pa.int64(), nullable=False),
        pa.field("execution_count", pa.int64(), nullable=False),
        pa.field("candidate_rate", pa.float64(), nullable=False),
    ]
)
TABLE_SCHEMAS: Final[Mapping[str, pa.Schema]] = {
    "outcomes": OUTCOMES_SCHEMA,
    "failures": FAILURES_SCHEMA,
    "candidate_multiplicity": MULTIPLICITY_SCHEMA,
    "candidate_origins": ORIGINS_SCHEMA,
    "evaluation_outcomes": EVALUATION_OUTCOMES_SCHEMA,
}


class PreprocessingAnalysisError(ValueError):
    """The immutable inputs cannot produce a trustworthy summary."""


@dataclass(frozen=True, slots=True)
class PreprocessingAnalysisArtifacts:
    output_dir: Path
    summary_path: Path
    manifest_path: Path
    table_paths: Mapping[str, Path]


def analyze_preprocessing_corpus(
    *,
    dataset_id: str,
    corpus_path: Path | str,
    run_dir: Path | str,
    output_dir: Path | str,
    candidate_evaluation: Path | str | None = None,
) -> PreprocessingAnalysisArtifacts:
    """Validate immutable inputs and publish compact derived artifacts."""
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"analysis output already exists: {destination}")
    try:
        with admitted_run_descriptor(
            label="analysis",
            dataset_id=dataset_id,
            corpus_path=corpus_path,
            preprocessing=run_dir,
            candidate_evaluation=candidate_evaluation,
        ) as descriptor:
            with staged_output_directory(destination) as temporary:
                table_root = temporary / "tables"
                table_root.mkdir()
                table_paths = {
                    name: table_root / f"{name}.parquet"
                    for name in TABLE_SCHEMAS
                }
                summary, table_rows = _summarize(
                    descriptor,
                    table_paths=table_paths,
                )
                table_manifest = {
                    name: {
                        "filename": f"tables/{path.name}",
                        "rows": table_rows[name],
                        "sha256": file_sha256(path),
                        "schema_hex": (
                            TABLE_SCHEMAS[name].serialize().to_pybytes().hex()
                        ),
                    }
                    for name, path in table_paths.items()
                }
                summary_path = temporary / SUMMARY_FILENAME
                _write_json(summary_path, summary)
                manifest = {
                    "schema_version": ANALYSIS_SCHEMA_VERSION,
                    "complete": True,
                    "inputs": _input_coordinates(descriptor),
                    "summary": {
                        "filename": SUMMARY_FILENAME,
                        "sha256": file_sha256(summary_path),
                    },
                    "tables": table_manifest,
                }
                _write_json(temporary / MANIFEST_FILENAME, manifest)
    except RunValidationError as exc:
        raise PreprocessingAnalysisError(str(exc)) from exc
    return PreprocessingAnalysisArtifacts(
        output_dir=destination,
        summary_path=destination / SUMMARY_FILENAME,
        manifest_path=destination / MANIFEST_FILENAME,
        table_paths={
            name: destination / "tables" / f"{name}.parquet"
            for name in TABLE_SCHEMAS
        },
    )


def _summarize(
    descriptor: RunDescriptor,
    *,
    table_paths: Mapping[str, Path],
) -> tuple[dict[str, object], dict[str, int]]:
    with tempfile.TemporaryDirectory(
        prefix="preprocessing-analysis-store-"
    ) as store_root:
        connection = sqlite3.connect(Path(store_root) / "analysis.sqlite3")
        try:
            connection.execute("PRAGMA temp_store = FILE")
            _create_aggregate_schema(connection)
            (
                sample_count,
                final_candidate_count,
                failure_count,
            ) = _load_preprocessing_aggregates(connection, descriptor)
            (
                evaluated_executions,
                evaluated_candidates,
            ) = _load_evaluation_aggregates(connection, descriptor)
            table_rows = _write_aggregate_tables(
                connection,
                table_paths=table_paths,
                sample_count=sample_count,
                evaluated_candidates=evaluated_candidates,
            )
            preprocessing_manifest = json.loads(
                descriptor.preprocessing_manifest_path.read_text(
                    encoding="utf-8"
                )
            )
            summary = {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "run": {
                    "run_id": descriptor.run_id,
                    "label": descriptor.label,
                    "dataset_id": descriptor.dataset_id,
                    "corpus_sha256": descriptor.corpus_sha256,
                    "preprocessing_manifest_sha256": (
                        descriptor.preprocessing_manifest_sha256
                    ),
                    "preprocessing_definition_identity": (
                        descriptor.definition_identity
                    ),
                    "candidate_evaluation_manifest_sha256": (
                        descriptor.evaluation_manifest_sha256
                    ),
                    "candidate_evaluation_generation_id": (
                        descriptor.evaluation_generation_id
                    ),
                    "candidate_evaluation_pointer_sha256": (
                        descriptor.evaluation_pointer_sha256
                    ),
                    "completed_at": preprocessing_manifest.get("completed_at"),
                },
                "counts": {
                    "samples": sample_count,
                    "final_candidates": final_candidate_count,
                    "failures": failure_count,
                    "evaluated_candidates": evaluated_candidates,
                    "evaluated_executions": evaluated_executions,
                },
            }
            return summary, table_rows
        finally:
            connection.close()


def _create_aggregate_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE outcomes(
            outcome TEXT PRIMARY KEY,
            sample_count INTEGER NOT NULL
        );
        CREATE TABLE multiplicity(
            candidate_count INTEGER PRIMARY KEY,
            sample_count INTEGER NOT NULL
        );
        CREATE TABLE failures(
            failure_code TEXT NOT NULL,
            failed_step TEXT NOT NULL,
            cause_key TEXT NOT NULL,
            cause TEXT,
            sample_count INTEGER NOT NULL,
            PRIMARY KEY(failure_code, failed_step, cause_key)
        );
        CREATE TABLE origins(
            path_json TEXT PRIMARY KEY,
            candidate_count INTEGER NOT NULL
        );
        CREATE TABLE evaluation_results(
            evaluation_key TEXT PRIMARY KEY,
            outcome TEXT NOT NULL
        );
        CREATE TABLE memberships(evaluation_key TEXT NOT NULL);
        """
    )


def _load_preprocessing_aggregates(
    connection: sqlite3.Connection,
    descriptor: RunDescriptor,
) -> tuple[int, int, int]:
    sample_count = 0
    final_candidate_count = 0
    failure_count = 0
    for row in _iter_rows(descriptor.results_path):
        sample_count += 1
        outcome = _required_text(row, "outcome")
        candidates = _required_nonnegative_int(row, "final_candidate_count")
        final_candidate_count += candidates
        _increment(
            connection,
            table="outcomes",
            key_columns=("outcome",),
            key_values=(outcome,),
            count_column="sample_count",
        )
        _increment(
            connection,
            table="multiplicity",
            key_columns=("candidate_count",),
            key_values=(candidates,),
            count_column="sample_count",
        )
        if row.get("failure_code") is not None:
            failure_count += 1
            failure_code = _required_text(row, "failure_code")
            failed_step = _required_text(row, "failed_step")
            cause = _optional_text(row.get("cause"))
            _increment(
                connection,
                table="failures",
                key_columns=(
                    "failure_code",
                    "failed_step",
                    "cause_key",
                    "cause",
                ),
                key_values=(
                    failure_code,
                    failed_step,
                    _canonical_json(cause),
                    cause,
                ),
                count_column="sample_count",
                conflict_columns=(
                    "failure_code",
                    "failed_step",
                    "cause_key",
                ),
            )
    for row in _iter_rows(descriptor.candidates_path, columns=("origins",)):
        for origin in normalize_origins(row.get("origins")):
            _increment(
                connection,
                table="origins",
                key_columns=("path_json",),
                key_values=(_canonical_json(origin),),
                count_column="candidate_count",
            )
    return sample_count, final_candidate_count, failure_count


def _load_evaluation_aggregates(
    connection: sqlite3.Connection,
    descriptor: RunDescriptor,
) -> tuple[int, int]:
    if descriptor.candidate_results_path is None:
        return 0, 0
    assert descriptor.candidate_membership_path is not None
    evaluated_executions = 0
    evaluated_candidates = 0
    for row in _iter_rows(
        descriptor.candidate_results_path,
        columns=("evaluation_key", "outcome"),
    ):
        evaluation_key = _required_text(row, "evaluation_key")
        outcome = row.get("outcome")
        try:
            connection.execute(
                "INSERT INTO evaluation_results VALUES (?, ?)",
                (
                    evaluation_key,
                    outcome if isinstance(outcome, str) else "<unmeasured>",
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise PreprocessingAnalysisError(
                "candidate evaluation results contain duplicate keys"
            ) from exc
        evaluated_executions += 1
    for row in _iter_rows(
        descriptor.candidate_membership_path,
        columns=("evaluation_key",),
    ):
        connection.execute(
            "INSERT INTO memberships VALUES (?)",
            (_required_text(row, "evaluation_key"),),
        )
        evaluated_candidates += 1
    missing = connection.execute(
        """
        SELECT 1
        FROM memberships
        LEFT JOIN evaluation_results USING(evaluation_key)
        WHERE evaluation_results.evaluation_key IS NULL
        LIMIT 1
        """
    ).fetchone()
    if missing is not None:
        raise PreprocessingAnalysisError(
            "candidate membership references a missing execution"
        )
    return evaluated_executions, evaluated_candidates


def _increment(
    connection: sqlite3.Connection,
    *,
    table: str,
    key_columns: Sequence[str],
    key_values: Sequence[object],
    count_column: str,
    conflict_columns: Sequence[str] | None = None,
) -> None:
    columns = (*key_columns, count_column)
    conflict = tuple(conflict_columns or key_columns)
    placeholders = ", ".join("?" for _ in columns)
    connection.execute(
        f"""
        INSERT INTO {table}({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT({", ".join(conflict)}) DO UPDATE SET
            {count_column} = {count_column} + 1
        """,
        (*key_values, 1),
    )


def _write_aggregate_tables(
    connection: sqlite3.Connection,
    *,
    table_paths: Mapping[str, Path],
    sample_count: int,
    evaluated_candidates: int,
) -> dict[str, int]:
    return {
        "outcomes": _write_aggregate_table(
            table_paths["outcomes"],
            connection=connection,
            schema=OUTCOMES_SCHEMA,
            query=(
                "SELECT outcome, sample_count FROM outcomes ORDER BY outcome"
            ),
            transform=lambda row: (
                row[0],
                row[1],
                cast(int, row[1]) / sample_count,
            ),
        ),
        "failures": _write_aggregate_table(
            table_paths["failures"],
            connection=connection,
            schema=FAILURES_SCHEMA,
            query=(
                "SELECT failure_code, failed_step, cause, sample_count "
                "FROM failures "
                "ORDER BY failure_code, failed_step, "
                "cause IS NOT NULL, cause"
            ),
        ),
        "candidate_multiplicity": _write_aggregate_table(
            table_paths["candidate_multiplicity"],
            connection=connection,
            schema=MULTIPLICITY_SCHEMA,
            query=(
                "SELECT candidate_count, sample_count "
                "FROM multiplicity ORDER BY candidate_count"
            ),
            transform=lambda row: (
                row[0],
                row[1],
                cast(int, row[1]) / sample_count,
            ),
        ),
        "candidate_origins": _write_aggregate_table(
            table_paths["candidate_origins"],
            connection=connection,
            schema=ORIGINS_SCHEMA,
            query=(
                "SELECT path_json, candidate_count "
                "FROM origins ORDER BY path_json"
            ),
        ),
        "evaluation_outcomes": _write_aggregate_table(
            table_paths["evaluation_outcomes"],
            connection=connection,
            schema=EVALUATION_OUTCOMES_SCHEMA,
            query=(
                """
                SELECT
                    evaluation_results.outcome,
                    count(memberships.evaluation_key),
                    execution_counts.execution_count
                FROM evaluation_results
                JOIN memberships USING(evaluation_key)
                JOIN (
                    SELECT outcome, count(*) AS execution_count
                    FROM evaluation_results
                    GROUP BY outcome
                ) AS execution_counts
                  ON execution_counts.outcome = evaluation_results.outcome
                GROUP BY
                    evaluation_results.outcome,
                    execution_counts.execution_count
                ORDER BY evaluation_results.outcome
                """
            ),
            transform=lambda row: (
                row[0],
                row[1],
                row[2],
                cast(int, row[1]) / evaluated_candidates,
            ),
        ),
    }


def _write_aggregate_table(
    path: Path,
    *,
    connection: sqlite3.Connection,
    schema: pa.Schema,
    query: str,
    transform: Callable[[Sequence[object]], Sequence[object]] | None = None,
) -> int:
    cursor = connection.execute(query)
    row_count = 0
    writer = pq.ParquetWriter(path, schema=schema, compression="zstd")
    try:
        while rows := cursor.fetchmany(_ROW_GROUP_SIZE):
            if transform is not None:
                rows = [tuple(transform(row)) for row in rows]
            table = pa.Table.from_pylist(
                [dict(zip(schema.names, row, strict=True)) for row in rows],
                schema=schema,
            )
            writer.write_table(table, row_group_size=_ROW_GROUP_SIZE)
            row_count += len(rows)
    finally:
        writer.close()
    return row_count


def _input_coordinates(descriptor: RunDescriptor) -> dict[str, object]:
    value: dict[str, object] = {
        "dataset": {"dataset_id": descriptor.dataset_id},
        "corpus": {
            "sha256": descriptor.corpus_sha256,
            "schema_hex": (
                pq.ParquetFile(descriptor.corpus_path)
                .schema_arrow.serialize()
                .to_pybytes()
                .hex()
            ),
        },
        "preprocessing": {
            "manifest_sha256": descriptor.preprocessing_manifest_sha256,
            "identity": descriptor.preprocessing_identity,
            "schema_version": descriptor.preprocessing_schema_version,
            "definition_identity": descriptor.definition_identity,
            "artifacts": {
                name: descriptor.artifact_sha256[name]
                for name in sorted(PROJECTED_ARTIFACT_SCHEMAS)
            },
        },
    }
    if descriptor.evaluation_manifest_sha256 is not None:
        value["candidate_evaluation"] = {
            "schema_version": CANDIDATE_EVALUATION_SCHEMA_VERSION,
            "generation_id": descriptor.evaluation_generation_id,
            "pointer_sha256": descriptor.evaluation_pointer_sha256,
            "manifest_sha256": descriptor.evaluation_manifest_sha256,
            "coordinates": thaw_json(
                descriptor.evaluation_coordinates
                if descriptor.evaluation_coordinates is not None
                else {}
            ),
        }
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _iter_rows(
    path: Path, *, columns: tuple[str, ...] | None = None
) -> Iterable[dict[str, object]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=_ROW_GROUP_SIZE,
        columns=list(columns) if columns is not None else None,
    ):
        yield from batch.to_pylist()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _required_text(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise PreprocessingAnalysisError(
            f"relation has invalid {field!r} value"
        )
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PreprocessingAnalysisError("expected text or null")
    return value


def _required_nonnegative_int(row: Mapping[str, object], field: str) -> int:
    value = row.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PreprocessingAnalysisError(
            f"relation has invalid {field!r} value"
        )
    return value


__all__ = (
    "ANALYSIS_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "PreprocessingAnalysisArtifacts",
    "PreprocessingAnalysisError",
    "TABLE_SCHEMAS",
    "analyze_preprocessing_corpus",
)
