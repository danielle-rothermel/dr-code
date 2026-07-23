"""Resumable HumanEval+ evaluation for completed preprocessing candidates.

The evaluator deliberately keeps mutable scheduling state in SQLite and emits
thin, deterministic Parquet projections.  Candidate membership is retained
independently from deduplicated execution work, so identical source can share
one execution without hiding where it appeared in the corpus.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import sqlite3
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq

from dr_code.humaneval.profiles import (
    DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
    HUMANEVAL_METRICS_PROFILE_ID,
    HUMANEVAL_METRICS_PROFILE_VERSION,
)
from dr_code.humaneval.batch_runner import runner_script
from dr_code.humaneval.subprocess_runner import (
    SubprocessError,
    SubprocessRunner,
    run_python_subprocess,
)
from dr_code.humaneval.sampling import load_human_eval_rows
from dr_code.humaneval.task import HumanEvalTask, parse_human_eval_dataset
from dr_code.preprocessing.steps.dedupe_candidates import (
    candidate_id_for_source,
)
from dr_code.metrics import (
    MetricName,
    MetricQuestion,
    MetricsDefinition,
    metrics_definition_hash,
)
from dr_code.metrics.engine.execution import (
    InMemoryExecutionCache,
    run_requests,
)
from dr_code.metrics.records import MetricRecord, RecordStatus
from dr_code.metrics.operators.code_test import (
    CodeTest,
    compute_code_test_result,
    plan_code_test_requests,
)
from dr_code.metrics.policy_example import derive_outcome


STATE_FILENAME: Final = "candidate_evaluation.sqlite3"
MANIFEST_FILENAME: Final = "candidate_evaluation_manifest.json"
MEMBERSHIP_FILENAME: Final = "candidate_membership.parquet"
RESULTS_FILENAME: Final = "candidate_results.parquet"
SCHEMA_VERSION: Final = 1
_LEASE_SECONDS: Final = 300.0
_PRODUCTION_PREFLIGHT_TIMEOUT_SECONDS: Final = 10.0
_DEFAULT_RUNNER_IDENTITY: Final = "subprocess:python-isolated@v1"

MEMBERSHIP_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("candidate_id", pa.string(), nullable=False),
        pa.field("candidate_index", pa.int64(), nullable=False),
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("source_kind", pa.string()),
        pa.field("source_sha256", pa.string(), nullable=False),
        pa.field("task_fingerprint", pa.string(), nullable=False),
        pa.field("evaluation_key", pa.string(), nullable=False),
        pa.field("metrics_profile", pa.string(), nullable=False),
        pa.field("operator", pa.string(), nullable=False),
    ]
)

RESULTS_SCHEMA: Final = pa.schema(
    [
        pa.field("evaluation_key", pa.string(), nullable=False),
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("cleaned_source", pa.string(), nullable=False),
        pa.field("source_sha256", pa.string(), nullable=False),
        pa.field("task_fingerprint", pa.string(), nullable=False),
        pa.field("metrics_profile", pa.string(), nullable=False),
        pa.field("operator", pa.string(), nullable=False),
        pa.field("record_status", pa.string(), nullable=False),
        pa.field("failure_type", pa.string()),
        pa.field("failure_message", pa.string()),
        pa.field("outcome", pa.string()),
        pa.field("function_count", pa.int64()),
        pa.field("best_function_name", pa.string()),
        pa.field("total_cases", pa.int64()),
        pa.field("passed_count", pa.int64()),
        pa.field("failed_count", pa.int64()),
        pa.field("error_count", pa.int64()),
        pa.field("timeout_count", pa.int64()),
        pa.field("coverage_complete", pa.bool_()),
    ]
)


class CandidateEvaluationError(ValueError):
    """The corpus, preprocessing run, state, or exports violate the contract."""


@dataclass(frozen=True, slots=True)
class EvaluationArtifacts:
    """The deterministic projections written by an evaluation run."""

    output_dir: Path
    membership_path: Path
    results_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class _Work:
    evaluation_key: str
    task_id: str
    source_sha256: str
    task_fingerprint: str
    candidate_source: str


@dataclass(frozen=True, slots=True)
class _ReuseSource:
    results_path: Path
    manifest_sha256: str
    results_sha256: str
    result_rows: int

    def descriptor(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "results_sha256": self.results_sha256,
            "result_rows": self.result_rows,
        }


_SEMANTIC_REUSE_COORDINATES: Final = (
    "snapshot_sha256",
    "metrics_definition",
    "metrics_definition_hash",
    "metrics_profile",
    "operator",
    "operator_settings",
    "runner_identity",
    "sandbox_image",
    "execution_fingerprint",
    "host_runtime",
)
_RESULT_VALUE_FIELDS: Final = (
    "function_count",
    "best_function_name",
    "total_cases",
    "passed_count",
    "failed_count",
    "error_count",
    "timeout_count",
    "coverage_complete",
)


def humaneval_metrics_definition() -> MetricsDefinition:
    """Return the pinned facts-first HumanEval metric declaration."""

    return MetricsDefinition(
        definition_id=HUMANEVAL_METRICS_PROFILE_ID,
        version=HUMANEVAL_METRICS_PROFILE_VERSION,
        questions=(
            MetricQuestion(
                metric=MetricName.CODE_TEST,
                on="candidate",
                settings={
                    "task_key": "task",
                    "timeout_seconds": DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
                },
            ),
        ),
    )


def evaluate_preprocessing_candidates(
    *,
    preprocessing_run: Path | str,
    corpus_path: Path | str,
    output_dir: Path | str,
    snapshot_path: Path | str,
    max_workers: int = 4,
    run_in_subprocess: SubprocessRunner | None = None,
    runner_identity: str | None = None,
    reuse_results_from: Sequence[Path | str] = (),
) -> EvaluationArtifacts:
    """Evaluate every candidate in a completed preprocessing run.

    The original corpus supplies the authoritative ``sample_id -> task_id``
    mapping.  ``snapshot_path`` is explicit so production can pin the exact
    HumanEval+ bytes while unit tests can provide a tiny validated snapshot.
    """
    if max_workers < 1:
        raise CandidateEvaluationError("max_workers must be at least 1")
    resolved_runner_identity = _resolve_runner_identity(
        run_in_subprocess, runner_identity
    )
    host_runtime = _host_runtime_coordinates()
    execution_fingerprint = _execution_fingerprint(
        resolved_runner_identity, host_runtime=host_runtime
    )
    runner = (
        run_python_subprocess
        if run_in_subprocess is None
        else run_in_subprocess
    )
    run_dir = Path(preprocessing_run).expanduser().resolve(strict=True)
    corpus_file = Path(corpus_path).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    snapshot_file = Path(snapshot_path).expanduser().resolve(strict=True)
    _validate_completed_preprocessing_run(run_dir)
    definition = humaneval_metrics_definition()
    base_immutable = _immutable_coordinates(
        run_dir=run_dir,
        corpus_file=corpus_file,
        snapshot_file=snapshot_file,
        definition=definition,
        runner_identity=resolved_runner_identity,
        host_runtime=host_runtime,
        execution_fingerprint=execution_fingerprint,
    )
    reuse_sources = _load_reuse_sources(
        reuse_results_from,
        destination=destination,
        expected_coordinates=base_immutable,
    )
    immutable = {
        **base_immutable,
        "reuse_result_sources": [
            source.descriptor() for source in reuse_sources
        ],
    }
    destination.mkdir(parents=True, exist_ok=True)
    connection = _open_state(destination / STATE_FILENAME)
    lease_id = uuid.uuid4().hex
    try:
        _initialize_state(connection, immutable)
        _acquire_lease(connection, lease_id)
        tasks = _load_tasks(snapshot_file)
        if run_in_subprocess is None:
            _preflight_production(tasks, run_in_subprocess=runner)
        task_fingerprints = {
            task_id: _task_fingerprint(task) for task_id, task in tasks.items()
        }
        _prepare_memberships(
            connection=connection,
            run_dir=run_dir,
            corpus_file=corpus_file,
            tasks=tasks,
            task_fingerprints=task_fingerprints,
            definition=definition,
            execution_fingerprint=execution_fingerprint,
        )
        _reset_stale_running(connection)
        _reuse_completed_results(
            connection,
            reuse_sources=reuse_sources,
            tasks=tasks,
            definition=definition,
            execution_fingerprint=execution_fingerprint,
        )
        _run_pending_work(
            connection=connection,
            lease_id=lease_id,
            tasks=tasks,
            definition=definition,
            max_workers=max_workers,
            run_in_subprocess=runner,
        )
        artifacts = EvaluationArtifacts(
            output_dir=destination,
            membership_path=destination / MEMBERSHIP_FILENAME,
            results_path=destination / RESULTS_FILENAME,
            manifest_path=destination / MANIFEST_FILENAME,
        )
        _export_artifacts(
            connection,
            artifacts,
            immutable,
            reuse_sources=reuse_sources,
        )
        return artifacts
    finally:
        _release_lease(connection, lease_id)
        connection.close()


def _validate_completed_preprocessing_run(run_dir: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    candidates_path = run_dir / "candidates.parquet"
    results_path = run_dir / "results.parquet"
    if not manifest_path.is_file() or not candidates_path.is_file():
        raise CandidateEvaluationError(
            "preprocessing run must contain manifest.json and candidates.parquet"
        )
    if not results_path.is_file():
        raise CandidateEvaluationError(
            "preprocessing run must contain results.parquet for membership validation"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CandidateEvaluationError(
            "preprocessing manifest is invalid JSON"
        ) from exc
    if manifest.get("complete") is not True:
        raise CandidateEvaluationError("preprocessing run is not complete")


def _immutable_coordinates(
    *,
    run_dir: Path,
    corpus_file: Path,
    snapshot_file: Path,
    definition: MetricsDefinition,
    runner_identity: str,
    host_runtime: Mapping[str, object],
    execution_fingerprint: str,
) -> dict[str, object]:
    operator_settings = definition.questions[0].settings
    return {
        "schema_version": SCHEMA_VERSION,
        "preprocessing_manifest_sha256": _sha256_file(
            run_dir / "manifest.json"
        ),
        "preprocessing_candidates_sha256": _sha256_file(
            run_dir / "candidates.parquet"
        ),
        "preprocessing_results_sha256": _sha256_file(
            run_dir / "results.parquet"
        ),
        "corpus_sha256": _sha256_file(corpus_file),
        "snapshot_sha256": _sha256_file(snapshot_file),
        "metrics_definition": definition.model_dump(mode="json"),
        "metrics_definition_hash": metrics_definition_hash(definition),
        "operator": "code_test@1",
        "operator_settings": operator_settings,
        "metrics_profile": (
            f"{HUMANEVAL_METRICS_PROFILE_ID}@{HUMANEVAL_METRICS_PROFILE_VERSION}"
        ),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "host_runtime": host_runtime,
        "trusted_source_sha256": _trusted_source_fingerprints(),
        # Retained as null for consumers of the legacy manifest shape.
        "sandbox_image": None,
        "runner_identity": runner_identity,
        "execution_fingerprint": execution_fingerprint,
    }


def _load_reuse_sources(
    paths: Sequence[Path | str],
    *,
    destination: Path,
    expected_coordinates: Mapping[str, object],
) -> tuple[_ReuseSource, ...]:
    sources: list[_ReuseSource] = []
    seen: set[tuple[str, str]] = set()
    for raw_path in paths:
        source_dir = Path(raw_path).expanduser().resolve(strict=True)
        if not source_dir.is_dir():
            raise CandidateEvaluationError(
                f"reuse source is not a directory: {source_dir}"
            )
        if source_dir == destination:
            raise CandidateEvaluationError(
                "evaluation output cannot reuse results from itself"
            )
        manifest_path = source_dir / MANIFEST_FILENAME
        results_path = source_dir / RESULTS_FILENAME
        if not manifest_path.is_file() or not results_path.is_file():
            raise CandidateEvaluationError(
                "reuse source must contain a candidate evaluation manifest "
                "and candidate results"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CandidateEvaluationError(
                f"reuse source manifest is invalid: {manifest_path}"
            ) from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("complete") is not True
        ):
            raise CandidateEvaluationError(
                f"reuse source is not complete: {manifest_path}"
            )
        for coordinate in _SEMANTIC_REUSE_COORDINATES:
            if coordinate not in manifest or _canonical_json(
                manifest[coordinate]
            ) != _canonical_json(expected_coordinates[coordinate]):
                raise CandidateEvaluationError(
                    "reuse source has incompatible semantic coordinate "
                    f"{coordinate!r}: {manifest_path}"
                )
        result_rows = manifest.get("result_rows")
        if not isinstance(result_rows, int) or result_rows < 0:
            raise CandidateEvaluationError(
                f"reuse source manifest has invalid result_rows: {manifest_path}"
            )
        try:
            parquet = pq.ParquetFile(results_path)
        except (OSError, pa.ArrowException) as exc:
            raise CandidateEvaluationError(
                f"reuse source results are invalid: {results_path}"
            ) from exc
        if not parquet.schema_arrow.equals(RESULTS_SCHEMA):
            raise CandidateEvaluationError(
                f"reuse source results schema is incompatible: {results_path}"
            )
        if parquet.metadata.num_rows != result_rows:
            raise CandidateEvaluationError(
                f"reuse source result row count does not match its manifest: {results_path}"
            )
        results_sha256 = _sha256_file(results_path)
        recorded_results_sha = manifest.get("candidate_results_sha256")
        if (
            not isinstance(recorded_results_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", recorded_results_sha) is None
        ):
            raise CandidateEvaluationError(
                "reuse source manifest must contain a valid "
                f"candidate_results_sha256: {manifest_path}"
            )
        if recorded_results_sha != results_sha256:
            raise CandidateEvaluationError(
                f"reuse source results hash does not match its manifest: {results_path}"
            )
        source = _ReuseSource(
            results_path=results_path,
            manifest_sha256=_sha256_file(manifest_path),
            results_sha256=results_sha256,
            result_rows=result_rows,
        )
        identity = (source.manifest_sha256, source.results_sha256)
        if identity in seen:
            raise CandidateEvaluationError(
                f"duplicate reuse source: {source_dir}"
            )
        seen.add(identity)
        sources.append(source)
    return tuple(sources)


def _load_tasks(snapshot_file: Path) -> dict[str, HumanEvalTask]:
    rows = load_human_eval_rows(snapshot_path=snapshot_file)
    tasks = parse_human_eval_dataset(rows)
    result = {task.task_id: task for task in tasks}
    if len(result) != len(tasks):
        raise CandidateEvaluationError(
            "snapshot contains duplicate task_id values"
        )
    return result


def _prepare_memberships(
    *,
    connection: sqlite3.Connection,
    run_dir: Path,
    corpus_file: Path,
    tasks: Mapping[str, HumanEvalTask],
    task_fingerprints: Mapping[str, str],
    definition: MetricsDefinition,
    execution_fingerprint: str,
) -> None:
    corpus = _load_corpus_metadata(corpus_file)
    expected_candidate_counts = _load_preprocessing_result_counts(
        run_dir / "results.parquet"
    )
    if set(corpus) != set(expected_candidate_counts):
        missing = sorted(set(corpus) - set(expected_candidate_counts))
        extra = sorted(set(expected_candidate_counts) - set(corpus))
        raise CandidateEvaluationError(
            "preprocessing result sample_id membership does not exactly match "
            f"the corpus (missing={missing[:3]!r}, extra={extra[:3]!r})"
        )
    candidates = pq.ParquetFile(run_dir / "candidates.parquet")
    required = {
        "sample_id",
        "candidate_id",
        "candidate_index",
        "cleaned_source",
        "source_sha256",
    }
    missing_columns = required - set(candidates.schema_arrow.names)
    if missing_columns:
        raise CandidateEvaluationError(
            "candidates artifact is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )
    connection.execute(
        """CREATE TEMP TABLE IF NOT EXISTS seen_input_membership (
            sample_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            candidate_index INTEGER NOT NULL,
            PRIMARY KEY (sample_id, candidate_id, candidate_index)
        )"""
    )
    connection.execute(
        """CREATE TEMP TABLE IF NOT EXISTS seen_candidate_id (
            sample_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            PRIMARY KEY (sample_id, candidate_id)
        )"""
    )
    connection.execute(
        """CREATE TEMP TABLE IF NOT EXISTS seen_candidate_index (
            sample_id TEXT NOT NULL,
            candidate_index INTEGER NOT NULL,
            PRIMARY KEY (sample_id, candidate_index)
        )"""
    )
    for batch in candidates.iter_batches(columns=sorted(required)):
        memberships: list[dict[str, object]] = []
        work: dict[str, _Work] = {}
        for row in pa.Table.from_batches([batch]).to_pylist():
            sample_id = _string(row["sample_id"], "candidate sample_id")
            candidate_id = _string(row["candidate_id"], "candidate_id")
            candidate_index = row["candidate_index"]
            candidate_source = _string(row["cleaned_source"], "cleaned_source")
            source_sha = _string(row["source_sha256"], "source_sha256")
            if not isinstance(candidate_index, int) or candidate_index < 0:
                raise CandidateEvaluationError(
                    "candidate_index must be a non-negative integer"
                )
            if (
                hashlib.sha256(candidate_source.encode("utf-8")).hexdigest()
                != source_sha
            ):
                raise CandidateEvaluationError(
                    f"candidate source_sha256 is invalid for {sample_id!r}/{candidate_id!r}"
                )
            if candidate_id != candidate_id_for_source(candidate_source):
                raise CandidateEvaluationError(
                    f"candidate_id is not content-derived for {sample_id!r}"
                )
            try:
                task_id, source_kind = corpus[sample_id]
            except KeyError as exc:
                raise CandidateEvaluationError(
                    f"candidate sample_id is absent from corpus: {sample_id!r}"
                ) from exc
            if not task_id or task_id not in tasks:
                raise CandidateEvaluationError(
                    f"corpus task_id is absent from the snapshot: {task_id!r}"
                )
            try:
                connection.execute(
                    """INSERT INTO seen_input_membership(
                        sample_id, candidate_id, candidate_index
                    ) VALUES (?, ?, ?)""",
                    (sample_id, candidate_id, candidate_index),
                )
            except sqlite3.IntegrityError:
                raise CandidateEvaluationError(
                    "candidates contain duplicate membership rows"
                ) from None
            for table, value, label in (
                ("seen_candidate_id", candidate_id, "candidate_id"),
                ("seen_candidate_index", candidate_index, "candidate_index"),
            ):
                try:
                    connection.execute(
                        f"INSERT INTO {table}(sample_id, {label}) VALUES (?, ?)",
                        (sample_id, value),
                    )
                except sqlite3.IntegrityError:
                    raise CandidateEvaluationError(
                        f"candidates contain duplicate {label} within a sample"
                    ) from None
            task_fingerprint = task_fingerprints[task_id]
            evaluation_key = _evaluation_key(
                task_id=task_id,
                task_fingerprint=task_fingerprint,
                candidate_source=candidate_source,
                definition=definition,
                execution_fingerprint=execution_fingerprint,
            )
            memberships.append(
                {
                    "sample_id": sample_id,
                    "candidate_id": candidate_id,
                    "candidate_index": candidate_index,
                    "task_id": task_id,
                    "source_kind": source_kind,
                    "source_sha256": source_sha,
                    "task_fingerprint": task_fingerprint,
                    "evaluation_key": evaluation_key,
                    "metrics_profile": (
                        f"{HUMANEVAL_METRICS_PROFILE_ID}@{HUMANEVAL_METRICS_PROFILE_VERSION}"
                    ),
                    "operator": "code_test@1",
                }
            )
            existing = work.get(evaluation_key)
            proposed = _Work(
                evaluation_key=evaluation_key,
                task_id=task_id,
                source_sha256=source_sha,
                task_fingerprint=task_fingerprint,
                candidate_source=candidate_source,
            )
            if (
                existing is not None
                and existing.candidate_source != candidate_source
            ):
                raise CandidateEvaluationError("evaluation key collision")
            work.setdefault(evaluation_key, proposed)
        _upsert_memberships_and_work(connection, memberships, work)
    _validate_candidate_counts(connection, expected_candidate_counts)


def _load_corpus_metadata(
    corpus_file: Path,
) -> dict[str, tuple[str | None, str | None]]:
    parquet = pq.ParquetFile(corpus_file)
    names = set(parquet.schema_arrow.names)
    required = {"sample_id", "task_id"}
    missing = required - names
    if missing:
        raise CandidateEvaluationError(
            "corpus is missing required column(s): "
            + ", ".join(sorted(missing))
        )
    source_column = "source_kind" if "source_kind" in names else None
    columns = ["sample_id", "task_id"] + (
        [source_column] if source_column else []
    )
    metadata: dict[str, tuple[str | None, str | None]] = {}
    for batch in parquet.iter_batches(columns=columns):
        for row in pa.Table.from_batches([batch]).to_pylist():
            sample_id = _string(row["sample_id"], "corpus sample_id")
            raw_task_id = row["task_id"]
            if raw_task_id is not None and not isinstance(raw_task_id, str):
                raise CandidateEvaluationError(
                    "corpus task_id must be a string when present"
                )
            task_id = raw_task_id
            source = row.get(source_column) if source_column else None
            if source is not None and not isinstance(source, str):
                raise CandidateEvaluationError(
                    "corpus source_kind must be a string when present"
                )
            if sample_id in metadata:
                raise CandidateEvaluationError(
                    "corpus contains duplicate sample_id values"
                )
            metadata[sample_id] = (task_id, source)
    return metadata


def _load_preprocessing_result_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    parquet = pq.ParquetFile(path)
    if "final_candidate_count" not in parquet.schema_arrow.names:
        raise CandidateEvaluationError(
            "preprocessing results are missing final_candidate_count"
        )
    for batch in parquet.iter_batches(
        columns=["sample_id", "final_candidate_count"]
    ):
        for row in pa.Table.from_batches([batch]).to_pylist():
            sample_id = _string(
                row["sample_id"], "preprocessing result sample_id"
            )
            count = row["final_candidate_count"]
            if not isinstance(count, int) or count < 0:
                raise CandidateEvaluationError(
                    "final_candidate_count must be a non-negative integer"
                )
            if sample_id in counts:
                raise CandidateEvaluationError(
                    "preprocessing results contain duplicate sample_id values"
                )
            counts[sample_id] = count
    return counts


def _validate_candidate_counts(
    connection: sqlite3.Connection, expected: Mapping[str, int]
) -> None:
    observed = {
        row[0]: (row[1], row[2])
        for row in connection.execute(
            """SELECT sample_id, COUNT(*), MAX(candidate_index)
               FROM seen_candidate_index GROUP BY sample_id"""
        )
    }
    for sample_id, expected_count in expected.items():
        row = observed.get(sample_id)
        actual_count = 0 if row is None else row[0]
        maximum_index = -1 if row is None else row[1]
        if (
            actual_count != expected_count
            or maximum_index != expected_count - 1
        ):
            raise CandidateEvaluationError(
                f"candidate rows do not match final_candidate_count for {sample_id!r}"
            )


def _open_state(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _initialize_state(
    connection: sqlite3.Connection, immutable: Mapping[str, object]
) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY NOT NULL,
            value_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS work (
            evaluation_key TEXT PRIMARY KEY NOT NULL,
            task_id TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            task_fingerprint TEXT NOT NULL,
            candidate_source TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending', 'running', 'completed')),
            record_status TEXT,
            failure_type TEXT,
            failure_message TEXT,
            values_json TEXT,
            completed_at TEXT,
            reused_from_manifest_sha256 TEXT
        );
        CREATE INDEX IF NOT EXISTS work_pending_claim
            ON work(status, evaluation_key);
        CREATE TABLE IF NOT EXISTS membership (
            sample_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            candidate_index INTEGER NOT NULL,
            task_id TEXT NOT NULL,
            source_kind TEXT,
            source_sha256 TEXT NOT NULL,
            task_fingerprint TEXT NOT NULL,
            evaluation_key TEXT NOT NULL REFERENCES work(evaluation_key),
            PRIMARY KEY (sample_id, candidate_id, candidate_index)
        );
        CREATE TABLE IF NOT EXISTS evaluator_lease (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            lease_id TEXT NOT NULL,
            heartbeat_at REAL NOT NULL
        );
        """
    )
    existing = dict(connection.execute("SELECT key, value_json FROM metadata"))
    encoded = {key: _canonical_json(value) for key, value in immutable.items()}
    if existing and existing != encoded:
        raise CandidateEvaluationError(
            "evaluation state is incompatible with the requested artifacts/settings"
        )
    if not existing:
        connection.executemany(
            "INSERT INTO metadata(key, value_json) VALUES (?, ?)",
            encoded.items(),
        )
    connection.commit()


def _upsert_memberships_and_work(
    connection: sqlite3.Connection,
    memberships: list[dict[str, object]],
    work: Mapping[str, _Work],
) -> None:
    with connection:
        for item in work.values():
            connection.execute(
                """
                INSERT INTO work(
                    evaluation_key, task_id, source_sha256, task_fingerprint,
                    candidate_source, status
                ) VALUES (?, ?, ?, ?, ?, 'pending')
                ON CONFLICT(evaluation_key) DO NOTHING
                """,
                (
                    item.evaluation_key,
                    item.task_id,
                    item.source_sha256,
                    item.task_fingerprint,
                    item.candidate_source,
                ),
            )
        for item in memberships:
            values = (
                item["sample_id"],
                item["candidate_id"],
                item["candidate_index"],
                item["task_id"],
                item["source_kind"],
                item["source_sha256"],
                item["task_fingerprint"],
                item["evaluation_key"],
            )
            existing = connection.execute(
                """SELECT task_id, source_kind, source_sha256, task_fingerprint,
                          evaluation_key FROM membership
                   WHERE sample_id = ? AND candidate_id = ? AND candidate_index = ?""",
                values[:3],
            ).fetchone()
            if existing is not None and existing != values[3:]:
                raise CandidateEvaluationError(
                    "persisted membership conflicts with input"
                )
            connection.execute(
                """INSERT INTO membership(
                    sample_id, candidate_id, candidate_index, task_id, source_kind,
                    source_sha256, task_fingerprint, evaluation_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sample_id, candidate_id, candidate_index) DO NOTHING""",
                values,
            )


def _reset_stale_running(connection: sqlite3.Connection) -> None:
    with connection:
        connection.execute(
            "UPDATE work SET status = 'pending' WHERE status = 'running'"
        )
        connection.execute(
            """UPDATE work
               SET status = 'pending', record_status = NULL,
                   failure_type = NULL, failure_message = NULL,
                   values_json = NULL, completed_at = NULL,
                   reused_from_manifest_sha256 = NULL
               WHERE status = 'completed'
                 AND record_status = 'infrastructure_failure'"""
        )


def _reuse_completed_results(
    connection: sqlite3.Connection,
    *,
    reuse_sources: Sequence[_ReuseSource],
    tasks: Mapping[str, HumanEvalTask],
    definition: MetricsDefinition,
    execution_fingerprint: str,
) -> None:
    if not reuse_sources:
        return
    task_fingerprints = {
        task_id: _task_fingerprint(task) for task_id, task in tasks.items()
    }
    with connection:
        for source in reuse_sources:
            parquet = pq.ParquetFile(source.results_path)
            for batch in parquet.iter_batches():
                for result in pa.Table.from_batches([batch]).to_pylist():
                    imported = _validated_reuse_result(
                        result,
                        task_fingerprints=task_fingerprints,
                        definition=definition,
                        execution_fingerprint=execution_fingerprint,
                        source_path=source.results_path,
                    )
                    if imported is None:
                        continue
                    (
                        evaluation_key,
                        task_id,
                        source_sha256,
                        task_fingerprint,
                        candidate_source,
                        record_status,
                        failure_type,
                        failure_message,
                        values_json,
                    ) = imported
                    target = connection.execute(
                        """SELECT task_id, source_sha256, task_fingerprint,
                                  candidate_source, status, record_status,
                                  failure_type, failure_message, values_json,
                                  reused_from_manifest_sha256
                             FROM work WHERE evaluation_key = ?""",
                        (evaluation_key,),
                    ).fetchone()
                    if target is None:
                        continue
                    expected_identity = (
                        task_id,
                        source_sha256,
                        task_fingerprint,
                        candidate_source,
                    )
                    if target[:4] != expected_identity:
                        raise CandidateEvaluationError(
                            "reuse result conflicts with target work identity "
                            f"for evaluation key {evaluation_key}"
                        )
                    if target[4] == "pending":
                        connection.execute(
                            """UPDATE work
                               SET status = 'completed', record_status = ?,
                                   failure_type = ?, failure_message = ?,
                                   values_json = ?, completed_at = ?,
                                   reused_from_manifest_sha256 = ?
                               WHERE evaluation_key = ? AND status = 'pending'""",
                            (
                                record_status,
                                failure_type,
                                failure_message,
                                values_json,
                                _timestamp(),
                                source.manifest_sha256,
                                evaluation_key,
                            ),
                        )
                    elif target[4] == "completed" and target[9] is not None:
                        if target[5:9] != (
                            record_status,
                            failure_type,
                            failure_message,
                            values_json,
                        ):
                            raise CandidateEvaluationError(
                                "reuse sources contain conflicting completed "
                                f"results for evaluation key {evaluation_key}"
                            )


def _validated_reuse_result(
    result: Mapping[str, object],
    *,
    task_fingerprints: Mapping[str, str],
    definition: MetricsDefinition,
    execution_fingerprint: str,
    source_path: Path,
) -> tuple[str, str, str, str, str, str, object, object, str] | None:
    evaluation_key = _string(
        result.get("evaluation_key"), "reuse evaluation_key"
    )
    task_id = _string(result.get("task_id"), "reuse task_id")
    candidate_source = _string(
        result.get("cleaned_source"), "reuse cleaned_source"
    )
    source_sha256 = _string(result.get("source_sha256"), "reuse source_sha256")
    task_fingerprint = _string(
        result.get("task_fingerprint"), "reuse task_fingerprint"
    )
    if _sha256_text(candidate_source) != source_sha256:
        raise CandidateEvaluationError(
            f"reuse result has invalid source hash: {source_path}"
        )
    if task_fingerprints.get(task_id) != task_fingerprint:
        raise CandidateEvaluationError(
            f"reuse result has invalid task fingerprint: {source_path}"
        )
    expected_key = _evaluation_key(
        task_id=task_id,
        task_fingerprint=task_fingerprint,
        candidate_source=candidate_source,
        definition=definition,
        execution_fingerprint=execution_fingerprint,
    )
    if evaluation_key != expected_key:
        raise CandidateEvaluationError(
            f"reuse result has invalid evaluation key: {source_path}"
        )
    expected_profile = (
        f"{HUMANEVAL_METRICS_PROFILE_ID}@{HUMANEVAL_METRICS_PROFILE_VERSION}"
    )
    if (
        result.get("metrics_profile") != expected_profile
        or result.get("operator") != "code_test@1"
    ):
        raise CandidateEvaluationError(
            f"reuse result has incompatible metric coordinates: {source_path}"
        )
    record_status = _string(result.get("record_status"), "reuse record_status")
    if record_status not in {"measured", "infrastructure_failure"}:
        raise CandidateEvaluationError(
            f"reuse result has invalid record status: {source_path}"
        )
    values = {field: result.get(field) for field in _RESULT_VALUE_FIELDS}
    if record_status != "measured":
        values = {}
    values_json = _canonical_json(values)
    reconstructed = _result_row(
        (
            evaluation_key,
            task_id,
            source_sha256,
            task_fingerprint,
            candidate_source,
            record_status,
            result.get("failure_type"),
            result.get("failure_message"),
            values_json if values else None,
        )
    )
    if any(
        reconstructed[field.name] != result.get(field.name)
        for field in RESULTS_SCHEMA
    ):
        raise CandidateEvaluationError(
            f"reuse result contains inconsistent completed fields: {source_path}"
        )
    if record_status != "measured":
        return None
    return (
        evaluation_key,
        task_id,
        source_sha256,
        task_fingerprint,
        candidate_source,
        record_status,
        result.get("failure_type"),
        result.get("failure_message"),
        values_json,
    )


def _acquire_lease(connection: sqlite3.Connection, lease_id: str) -> None:
    """Claim the mutable state, recovering only an expired owner lease.

    A separate process with a fresh lease is not a stale resume.  Treating it
    as one would reset its running rows and permit duplicate subprocess work.
    """
    now = time.time()
    with connection:
        row = connection.execute(
            "SELECT lease_id, heartbeat_at FROM evaluator_lease WHERE singleton = 1"
        ).fetchone()
        if row is not None and now - row[1] < _LEASE_SECONDS:
            raise CandidateEvaluationError(
                "candidate evaluation state is currently owned by a live evaluator"
            )
        connection.execute("DELETE FROM evaluator_lease WHERE singleton = 1")
        connection.execute(
            """INSERT INTO evaluator_lease(singleton, lease_id, heartbeat_at)
               VALUES (1, ?, ?)""",
            (lease_id, now),
        )


def _heartbeat(connection: sqlite3.Connection, lease_id: str) -> None:
    with connection:
        updated = connection.execute(
            """UPDATE evaluator_lease SET heartbeat_at = ?
               WHERE singleton = 1 AND lease_id = ?""",
            (time.time(), lease_id),
        )
        if updated.rowcount != 1:
            raise CandidateEvaluationError(
                "candidate evaluation lease was lost"
            )


def _release_lease(connection: sqlite3.Connection, lease_id: str) -> None:
    try:
        with connection:
            connection.execute(
                "DELETE FROM evaluator_lease WHERE singleton = 1 AND lease_id = ?",
                (lease_id,),
            )
    except sqlite3.Error:
        # Preserve the primary failure.  A remaining lease naturally expires.
        pass


def _run_pending_work(
    *,
    connection: sqlite3.Connection,
    lease_id: str,
    tasks: Mapping[str, HumanEvalTask],
    definition: MetricsDefinition,
    max_workers: int,
    run_in_subprocess: SubprocessRunner,
) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[
            Future[tuple[str, MetricRecord | None, SubprocessError | None]],
            str,
        ] = {}
        while True:
            while len(futures) < max_workers:
                work = _claim_next_work(connection)
                if work is None:
                    break
                future = executor.submit(
                    _measure_work,
                    work,
                    tasks[work.task_id],
                    definition,
                    run_in_subprocess,
                )
                futures[future] = work.evaluation_key
            if not futures:
                return
            completed, _ = wait(
                futures,
                timeout=_LEASE_SECONDS / 3,
                return_when=FIRST_COMPLETED,
            )
            if not completed:
                _heartbeat(connection, lease_id)
                continue
            for future in completed:
                futures.pop(future)
                key, record, infrastructure_failure = future.result()
                _complete_work(connection, key, record, infrastructure_failure)
            _heartbeat(connection, lease_id)


def _claim_next_work(connection: sqlite3.Connection) -> _Work | None:
    with connection:
        row = connection.execute(
            """SELECT evaluation_key, task_id, source_sha256,
                      task_fingerprint, candidate_source
                 FROM work WHERE status = 'pending' ORDER BY evaluation_key LIMIT 1"""
        ).fetchone()
        if row is None:
            return None
        claimed = connection.execute(
            "UPDATE work SET status = 'running' WHERE evaluation_key = ? AND status = 'pending'",
            (row[0],),
        )
        if claimed.rowcount != 1:
            return None
        return _Work(*row)


def _measure_work(
    work: _Work,
    task: HumanEvalTask,
    definition: MetricsDefinition,
    run_in_subprocess: SubprocessRunner,
) -> tuple[str, MetricRecord | None, SubprocessError | None]:
    try:
        requests = plan_code_test_requests(
            task=task,
            candidate_source=work.candidate_source,
            timeout_seconds=DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
        )
        outcomes = run_requests(
            requests,
            run_in_subprocess=run_in_subprocess,
            cache=InMemoryExecutionCache(),
        )
        result = compute_code_test_result(
            task=task,
            candidate_source=work.candidate_source,
            timeout_seconds=DEFAULT_HUMANEVAL_TIMEOUT_SECONDS,
            outcomes=outcomes,
        )
        record = MetricRecord(
            metric=MetricName.CODE_TEST,
            metric_version=CodeTest.VERSION,
            settings=definition.questions[0].settings,
            on_key="candidate",
            producer_id="external",
            producer_version=None,
            producer_definition_hash=None,
            metrics_definition_id=definition.definition_id,
            metrics_definition_version=definition.version,
            status=RecordStatus.MEASURED,
            values=result.to_values(),
        )
        return work.evaluation_key, record, None
    except SubprocessError as exc:
        return work.evaluation_key, None, exc


def _complete_work(
    connection: sqlite3.Connection,
    evaluation_key: str,
    record: MetricRecord | None,
    infrastructure_failure: SubprocessError | None,
) -> None:
    if infrastructure_failure is not None:
        fields = (
            "infrastructure_failure",
            type(infrastructure_failure).__name__,
            str(infrastructure_failure),
            None,
        )
    else:
        assert record is not None
        fields = (
            str(record.status),
            record.failure_type,
            record.failure_message,
            _canonical_json(record.values),
        )
    with connection:
        completed = connection.execute(
            """UPDATE work
               SET status = 'completed', record_status = ?, failure_type = ?,
                   failure_message = ?, values_json = ?, completed_at = ?
               WHERE evaluation_key = ? AND status = 'running'""",
            (*fields, _timestamp(), evaluation_key),
        )
        if completed.rowcount != 1:
            raise CandidateEvaluationError(
                "lost ownership while completing evaluation work"
            )


def _export_artifacts(
    connection: sqlite3.Connection,
    artifacts: EvaluationArtifacts,
    immutable: Mapping[str, object],
    *,
    reuse_sources: Sequence[_ReuseSource],
) -> None:
    outstanding = connection.execute(
        "SELECT COUNT(*) FROM work WHERE status != 'completed'"
    ).fetchone()[0]
    if outstanding:
        raise CandidateEvaluationError(
            "cannot export incomplete evaluation state"
        )
    membership_count = connection.execute(
        "SELECT COUNT(*) FROM membership"
    ).fetchone()[0]
    result_count = connection.execute("SELECT COUNT(*) FROM work").fetchone()[
        0
    ]
    reused_counts = dict(
        connection.execute(
            """SELECT reused_from_manifest_sha256, COUNT(*)
                 FROM work
                WHERE reused_from_manifest_sha256 IS NOT NULL
                GROUP BY reused_from_manifest_sha256"""
        )
    )
    _atomic_write_query_parquet(
        artifacts.membership_path,
        MEMBERSHIP_SCHEMA,
        connection.execute(
            """SELECT sample_id, candidate_id, candidate_index, task_id, source_kind,
                      source_sha256, task_fingerprint, evaluation_key
               FROM membership ORDER BY sample_id, candidate_index, candidate_id"""
        ),
        _membership_row,
    )
    _atomic_write_query_parquet(
        artifacts.results_path,
        RESULTS_SCHEMA,
        connection.execute(
            """SELECT evaluation_key, task_id, source_sha256,
                      task_fingerprint, candidate_source, record_status,
                      failure_type, failure_message, values_json
               FROM work ORDER BY evaluation_key"""
        ),
        _result_row,
    )
    reuse_provenance = [
        {
            **source.descriptor(),
            "reused_result_rows": reused_counts.get(source.manifest_sha256, 0),
        }
        for source in reuse_sources
    ]
    manifest = {
        **immutable,
        "membership_rows": membership_count,
        "result_rows": result_count,
        "candidate_membership_sha256": _sha256_file(artifacts.membership_path),
        "candidate_results_sha256": _sha256_file(artifacts.results_path),
        "reused_result_rows": sum(reused_counts.values()),
        "reused_result_rows_by_source": reuse_provenance,
        "complete": True,
        "completed_at": _timestamp(),
    }
    _atomic_write_text(
        artifacts.manifest_path, _canonical_json(manifest) + "\n"
    )


def _result_row(row: tuple[object, ...]) -> dict[str, object]:
    values_json = row[8]
    values = json.loads(values_json) if isinstance(values_json, str) else {}
    assert isinstance(values, dict)
    outcome = None
    if row[5] == "measured":
        record = MetricRecord(
            metric=MetricName.CODE_TEST,
            metric_version=CodeTest.VERSION,
            settings=humaneval_metrics_definition().questions[0].settings,
            on_key="candidate",
            producer_id="external",
            producer_version=None,
            producer_definition_hash=None,
            metrics_definition_id=HUMANEVAL_METRICS_PROFILE_ID,
            metrics_definition_version=HUMANEVAL_METRICS_PROFILE_VERSION,
            status=RecordStatus.MEASURED,
            values=values,
        )
        outcome = str(derive_outcome(record))
    return {
        "evaluation_key": row[0],
        "task_id": row[1],
        "cleaned_source": row[4],
        "source_sha256": row[2],
        "task_fingerprint": row[3],
        "metrics_profile": f"{HUMANEVAL_METRICS_PROFILE_ID}@{HUMANEVAL_METRICS_PROFILE_VERSION}",
        "operator": "code_test@1",
        "record_status": row[5],
        "failure_type": row[6],
        "failure_message": row[7],
        "outcome": outcome,
        "function_count": values.get("function_count"),
        "best_function_name": values.get("best_function_name"),
        "total_cases": values.get("total_cases"),
        "passed_count": values.get("passed_count"),
        "failed_count": values.get("failed_count"),
        "error_count": values.get("error_count"),
        "timeout_count": values.get("timeout_count"),
        "coverage_complete": values.get("coverage_complete"),
    }


def _membership_row(row: tuple[object, ...]) -> dict[str, object]:
    return {
        "sample_id": row[0],
        "candidate_id": row[1],
        "candidate_index": row[2],
        "task_id": row[3],
        "source_kind": row[4],
        "source_sha256": row[5],
        "task_fingerprint": row[6],
        "evaluation_key": row[7],
        "metrics_profile": (
            f"{HUMANEVAL_METRICS_PROFILE_ID}@{HUMANEVAL_METRICS_PROFILE_VERSION}"
        ),
        "operator": "code_test@1",
    }


def _task_fingerprint(task: HumanEvalTask) -> str:
    payload = task.model_dump(mode="json")
    return _sha256_text(_canonical_json(payload))


def _evaluation_key(
    *,
    task_id: str,
    task_fingerprint: str,
    candidate_source: str,
    definition: MetricsDefinition,
    execution_fingerprint: str,
) -> str:
    return _sha256_text(
        _canonical_json(
            {
                "task_id": task_id,
                "task_fingerprint": task_fingerprint,
                "candidate_source_sha256": _sha256_text(candidate_source),
                "metrics_definition_hash": metrics_definition_hash(definition),
                "metrics_profile": f"{HUMANEVAL_METRICS_PROFILE_ID}@{HUMANEVAL_METRICS_PROFILE_VERSION}",
                "operator": "code_test@1",
                "settings": definition.questions[0].settings,
                "execution_fingerprint": execution_fingerprint,
            }
        )
    )


def _execution_fingerprint(
    runner_identity: str,
    *,
    host_runtime: Mapping[str, object] | None = None,
) -> str:
    """Hash every trusted program/runtime coordinate affecting execution."""

    resolved_host_runtime = (
        _host_runtime_coordinates() if host_runtime is None else host_runtime
    )
    return _sha256_text(
        _canonical_json(
            {
                "trusted_source_sha256": _trusted_source_fingerprints(),
                "python": sys.version,
                "host_runtime": resolved_host_runtime,
                "sandbox_image": None,
                "runner_identity": runner_identity,
            }
        )
    )


def _host_runtime_coordinates() -> dict[str, object]:
    installed_distributions = _installed_distributions()
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python_executable_sha256": _sha256_file(
            Path(sys.executable).resolve(strict=True)
        ),
        "installed_distributions": installed_distributions,
        "installed_distributions_sha256": _sha256_text(
            _canonical_json(installed_distributions)
        ),
    }


def _installed_distributions() -> list[dict[str, str]]:
    coordinates: set[tuple[str, str]] = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        version = distribution.version
        if not name or not version:
            raise CandidateEvaluationError(
                "installed distribution is missing its name or version"
            )
        normalized_name = re.sub(r"[-_.]+", "-", name).lower()
        coordinates.add((normalized_name, version))
    return [
        {"name": name, "version": version}
        for name, version in sorted(coordinates)
    ]


def _resolve_runner_identity(
    run_in_subprocess: SubprocessRunner | None, runner_identity: str | None
) -> str:
    if run_in_subprocess is None:
        if runner_identity is not None:
            raise CandidateEvaluationError(
                "runner_identity is only valid with an injected subprocess runner"
            )
        return _DEFAULT_RUNNER_IDENTITY
    if not runner_identity or not runner_identity.strip():
        raise CandidateEvaluationError(
            "injected run_in_subprocess requires a non-empty runner_identity"
        )
    return runner_identity


def _trusted_source_fingerprints() -> dict[str, str]:
    modules = (
        "dr_code.corpus.candidate_evaluation",
        "dr_code.humaneval.subprocess_runner",
        "dr_code.humaneval.task",
        "dr_code.humaneval.parsed_tests",
        "dr_code.humaneval.parsed_code",
        "dr_code.humaneval.batch_runner",
        "dr_code.metrics.engine.execution",
        "dr_code.metrics.operators.code_test",
    )
    fingerprints: dict[str, str] = {
        "runner_script": _sha256_text(runner_script())
    }
    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module.__file__ is not None
        fingerprints[module_name] = _sha256_file(Path(module.__file__))
    return fingerprints


def _preflight_production(
    tasks: Mapping[str, HumanEvalTask],
    *,
    run_in_subprocess: SubprocessRunner = run_python_subprocess,
) -> None:
    """Fail closed if local isolated Python cannot run trusted benchmark code."""

    numpy_probe = run_in_subprocess(
        source="import numpy\nprint(numpy.__version__)\n",
        input_json="{}",
        timeout_seconds=_PRODUCTION_PREFLIGHT_TIMEOUT_SECONDS,
    )
    if numpy_probe.returncode != 0:
        raise CandidateEvaluationError(
            "production subprocess preflight could not import NumPy"
        )
    for task in tasks.values():
        requests = plan_code_test_requests(
            task=task,
            candidate_source=task.ground_truth_code,
            timeout_seconds=_PRODUCTION_PREFLIGHT_TIMEOUT_SECONDS,
        )
        outcomes = run_requests(
            requests,
            run_in_subprocess=run_in_subprocess,
            cache=InMemoryExecutionCache(),
        )
        result = compute_code_test_result(
            task=task,
            candidate_source=task.ground_truth_code,
            timeout_seconds=_PRODUCTION_PREFLIGHT_TIMEOUT_SECONDS,
            outcomes=outcomes,
        )
        if (
            result.passed_count != result.total_cases
            or result.error_count
            or result.failed_count
            or result.timeout_count
            or not result.coverage_complete
        ):
            raise CandidateEvaluationError(
                "production subprocess preflight failed for "
                f"{task.task_id}: passed={result.passed_count}, "
                f"failed={result.failed_count}, errors={result.error_count}, "
                f"timeouts={result.timeout_count}, "
                f"coverage_complete={result.coverage_complete}"
            )


def _atomic_write_parquet(
    path: Path, rows: list[dict[str, object]], schema: pa.Schema
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, temporary, compression="zstd", use_dictionary=False)
    os.replace(temporary, path)


def _atomic_write_query_parquet(
    path: Path,
    schema: pa.Schema,
    cursor: sqlite3.Cursor,
    row_mapper: Callable[[tuple[object, ...]], dict[str, object]],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with pq.ParquetWriter(
        temporary, schema, compression="zstd", use_dictionary=False
    ) as writer:
        while rows := cursor.fetchmany(1_000):
            writer.write_table(
                pa.Table.from_pylist(
                    [row_mapper(row) for row in rows], schema=schema
                )
            )
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CandidateEvaluationError(f"{field} must be a non-empty string")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()
