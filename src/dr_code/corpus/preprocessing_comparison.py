"""Deterministic identity-level comparison of immutable preprocessing runs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, cast

import pyarrow as pa
import pyarrow.parquet as pq

from dr_code.corpus.atomic_directory import staged_output_directory
from dr_code.corpus.run_descriptor import (
    RunDescriptor,
    RunValidationError,
    admitted_run_descriptor,
    normalize_origins,
    thaw_json,
)


COMPARISON_SCHEMA_VERSION: Final = 1
SUMMARY_FILENAME: Final = "comparison_summary.json"
MANIFEST_FILENAME: Final = "comparison_manifest.json"
_SEMANTIC_RESULT_FIELDS: Final = (
    "raw_output_sha256",
    "decoder_output_presence",
    "outcome",
    "outcome_code",
    "failure_code",
    "failed_step",
    "cause",
    "propagated_through",
    "final_candidate_count",
)
_INPUT_BATCH_SIZE: Final = 65_536
_OUTPUT_BATCH_SIZE: Final = 10_000

SAMPLE_TRANSITIONS_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("before_output_sha256", pa.string()),
        pa.field("after_output_sha256", pa.string()),
        pa.field("before_output_presence", pa.string(), nullable=False),
        pa.field("after_output_presence", pa.string(), nullable=False),
        pa.field("before_outcome", pa.string(), nullable=False),
        pa.field("after_outcome", pa.string(), nullable=False),
        pa.field("before_outcome_code", pa.string()),
        pa.field("after_outcome_code", pa.string()),
        pa.field("before_failure_code", pa.string()),
        pa.field("after_failure_code", pa.string()),
        pa.field("before_failed_step", pa.string()),
        pa.field("after_failed_step", pa.string()),
        pa.field("before_cause", pa.string()),
        pa.field("after_cause", pa.string()),
        pa.field("before_propagated_through", pa.list_(pa.string())),
        pa.field("after_propagated_through", pa.list_(pa.string())),
        pa.field("before_final_candidate_count", pa.int64(), nullable=False),
        pa.field("after_final_candidate_count", pa.int64(), nullable=False),
        pa.field("output_identity_changed", pa.bool_(), nullable=False),
        pa.field("outcome_changed", pa.bool_(), nullable=False),
        pa.field("semantic_result_changed", pa.bool_(), nullable=False),
        pa.field("changed_fields", pa.list_(pa.string()), nullable=False),
        pa.field("change", pa.string(), nullable=False),
    ]
)

CANDIDATE_CHANGES_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("candidate_id", pa.string(), nullable=False),
        pa.field("before_present", pa.bool_(), nullable=False),
        pa.field("after_present", pa.bool_(), nullable=False),
        pa.field("before_candidate_index", pa.int64()),
        pa.field("after_candidate_index", pa.int64()),
        pa.field("before_source_sha256", pa.string()),
        pa.field("after_source_sha256", pa.string()),
        pa.field("membership_changed", pa.bool_(), nullable=False),
        pa.field("source_changed", pa.bool_(), nullable=False),
        pa.field("change", pa.string(), nullable=False),
    ]
)

PROVENANCE_PATH_DELTAS_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("candidate_id", pa.string(), nullable=False),
        pa.field("path_json", pa.string(), nullable=False),
        pa.field("before_count", pa.int64(), nullable=False),
        pa.field("after_count", pa.int64(), nullable=False),
        pa.field("count_delta", pa.int64(), nullable=False),
        pa.field("change", pa.string(), nullable=False),
    ]
)

EVALUATION_MEMBERSHIP_CHANGES_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("candidate_id", pa.string(), nullable=False),
        pa.field("before_present", pa.bool_(), nullable=False),
        pa.field("after_present", pa.bool_(), nullable=False),
        pa.field("before_evaluation_key", pa.string()),
        pa.field("after_evaluation_key", pa.string()),
        pa.field("before_membership_json", pa.string()),
        pa.field("after_membership_json", pa.string()),
        pa.field("change", pa.string(), nullable=False),
    ]
)

EVALUATION_RESULT_CHANGES_SCHEMA: Final = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("candidate_id", pa.string(), nullable=False),
        pa.field("before_evaluation_key", pa.string()),
        pa.field("after_evaluation_key", pa.string()),
        pa.field("before_result_json", pa.string()),
        pa.field("after_result_json", pa.string()),
        pa.field("change", pa.string(), nullable=False),
    ]
)

_RELATION_SCHEMAS: Final[Mapping[str, pa.Schema]] = {
    "sample_outcome_transitions": SAMPLE_TRANSITIONS_SCHEMA,
    "candidate_changes": CANDIDATE_CHANGES_SCHEMA,
    "provenance_path_deltas": PROVENANCE_PATH_DELTAS_SCHEMA,
    "evaluation_membership_changes": EVALUATION_MEMBERSHIP_CHANGES_SCHEMA,
    "evaluation_result_changes": EVALUATION_RESULT_CHANGES_SCHEMA,
}


class PreprocessingComparisonError(ValueError):
    """The compared bundles cannot produce a trustworthy identity audit."""


@dataclass(frozen=True, slots=True)
class PreprocessingComparisonArtifacts:
    """The append-only files emitted for one before/after comparison."""

    output_dir: Path
    summary_path: Path
    manifest_path: Path
    relation_paths: Mapping[str, Path]


@dataclass(slots=True)
class _StreamingSummary:
    corpus_rows: int
    evaluation_included: bool
    changes: dict[str, Counter[str]] = field(default_factory=dict)
    row_counts: Counter[str] = field(default_factory=Counter)
    before_presence: Counter[str] = field(default_factory=Counter)
    after_presence: Counter[str] = field(default_factory=Counter)
    output_identity_changed_count: int = 0
    outcome_changed_count: int = 0
    semantic_result_changed_count: int = 0
    transitions: Counter[tuple[str, str]] = field(default_factory=Counter)
    provenance_before_count: int = 0
    provenance_after_count: int = 0
    provenance_net_count_delta: int = 0

    def consume(self, relation: str, row: Mapping[str, object]) -> None:
        self.row_counts[relation] += 1
        self.changes.setdefault(relation, Counter())[
            cast(str, row["change"])
        ] += 1
        if relation == "sample_outcome_transitions":
            self.output_identity_changed_count += (
                row["output_identity_changed"] is True
            )
            self.outcome_changed_count += row["outcome_changed"] is True
            self.semantic_result_changed_count += (
                row["semantic_result_changed"] is True
            )
            self.transitions[
                (
                    cast(str, row["before_outcome"]),
                    cast(str, row["after_outcome"]),
                )
            ] += 1
        if relation in {
            "candidate_changes",
            "evaluation_membership_changes",
        }:
            self.before_presence[relation] += row["before_present"] is True
            self.after_presence[relation] += row["after_present"] is True
        if relation == "provenance_path_deltas":
            self.provenance_before_count += cast(int, row["before_count"])
            self.provenance_after_count += cast(int, row["after_count"])
            self.provenance_net_count_delta += cast(int, row["count_delta"])

    def value(
        self, before: RunDescriptor, after: RunDescriptor
    ) -> dict[str, object]:
        samples = "sample_outcome_transitions"
        candidates = "candidate_changes"
        provenance = "provenance_path_deltas"
        memberships = "evaluation_membership_changes"
        results = "evaluation_result_changes"
        return {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "dataset_id": before.dataset_id,
            "corpus_rows": self.corpus_rows,
            samples: {
                **self._change_counts(samples),
                "output_identity_changed_count": (
                    self.output_identity_changed_count
                ),
                "outcome_changed_count": self.outcome_changed_count,
                "semantic_result_changed_count": (
                    self.semantic_result_changed_count
                ),
                "transitions": [
                    {
                        "before_outcome": old,
                        "after_outcome": new,
                        "count": count,
                    }
                    for (old, new), count in sorted(self.transitions.items())
                ],
            },
            candidates: self._presence_counts(candidates),
            provenance: {
                **self._change_counts(provenance),
                "before_count": self.provenance_before_count,
                "after_count": self.provenance_after_count,
                "net_count_delta": self.provenance_net_count_delta,
            },
            "evaluation": {
                "included": self.evaluation_included,
                "membership_changes": self._presence_counts(memberships),
                "result_changes": self._change_counts(results),
                "coordinates": _evaluation_coordinate_summary(before, after),
            },
            "reconciliation": {
                "sample_identity_rows": self.row_counts[samples],
                "sample_rows_match_corpus": (
                    self.row_counts[samples] == self.corpus_rows
                ),
                "candidate_before_count": self.before_presence[candidates],
                "candidate_after_count": self.after_presence[candidates],
                "provenance_before_count": self.provenance_before_count,
                "provenance_after_count": self.provenance_after_count,
                "evaluation_membership_before_count": (
                    self.before_presence[memberships]
                ),
                "evaluation_membership_after_count": (
                    self.after_presence[memberships]
                ),
                "evaluation_result_identity_rows": self.row_counts[results],
            },
        }

    def _change_counts(self, relation: str) -> dict[str, object]:
        counts = self.changes.get(relation, Counter())
        rows = self.row_counts[relation]
        return {
            "identity_rows": rows,
            "changed_identity_rows": rows - counts.get("unchanged", 0),
            "by_change": dict(sorted(counts.items())),
        }

    def _presence_counts(self, relation: str) -> dict[str, object]:
        before = self.before_presence[relation]
        after = self.after_presence[relation]
        return {
            **self._change_counts(relation),
            "before_count": before,
            "after_count": after,
            "count_delta": after - before,
        }


def compare_preprocessing_runs(
    *,
    dataset_id: str,
    corpus_path: Path | str,
    before_run: Path | str,
    after_run: Path | str,
    output_dir: Path | str,
    before_evaluation: Path | str | None = None,
    after_evaluation: Path | str | None = None,
) -> PreprocessingComparisonArtifacts:
    """Write a deterministic audit of two immutable runs over one corpus."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"comparison output already exists: {destination}"
        )
    if (before_evaluation is None) != (after_evaluation is None):
        raise PreprocessingComparisonError(
            "before and after evaluation bundles must be supplied together"
        )
    corpus = Path(corpus_path).expanduser().resolve(strict=True)
    with (
        _load_descriptor(
            label="before",
            dataset_id=dataset_id,
            corpus=corpus,
            preprocessing=before_run,
            evaluation=before_evaluation,
        ) as before_descriptor,
        _load_descriptor(
            label="after",
            dataset_id=dataset_id,
            corpus=corpus,
            preprocessing=after_run,
            evaluation=after_evaluation,
        ) as after_descriptor,
    ):
        if before_descriptor.corpus_sha256 != after_descriptor.corpus_sha256:
            raise PreprocessingComparisonError(
                "before and after preprocessing runs reference different corpora"
            )
        return _compare_admitted_runs(
            destination=destination,
            before_descriptor=before_descriptor,
            after_descriptor=after_descriptor,
        )


def _compare_admitted_runs(
    *,
    destination: Path,
    before_descriptor: RunDescriptor,
    after_descriptor: RunDescriptor,
) -> PreprocessingComparisonArtifacts:
    with staged_output_directory(destination) as temporary:
        with _comparison_store(temporary) as connection:
            corpus_rows = _load_corpus(
                connection, before_descriptor.corpus_path
            )
            _load_run(connection, "before", before_descriptor, corpus_rows)
            _load_run(connection, "after", after_descriptor, corpus_rows)
            streaming_summary = _StreamingSummary(
                corpus_rows=corpus_rows,
                evaluation_included=before_descriptor.has_evaluation,
            )
            relation_manifest: dict[str, object] = {}
            for name, schema in _RELATION_SCHEMAS.items():
                path = temporary / f"{name}.parquet"
                row_count = _write_relation(
                    path,
                    _comparison_rows(connection, name),
                    schema,
                    lambda row, relation=name: streaming_summary.consume(
                        relation, row
                    ),
                )
                relation_manifest[name] = {
                    "filename": path.name,
                    "row_count": row_count,
                    "sha256": _sha256_file(path),
                    "schema": schema.serialize().to_pybytes().hex(),
                }
            summary = streaming_summary.value(
                before_descriptor, after_descriptor
            )

        temporary_summary = temporary / SUMMARY_FILENAME
        _write_json(temporary_summary, summary)
        _write_json(
            temporary / MANIFEST_FILENAME,
            {
                "schema_version": COMPARISON_SCHEMA_VERSION,
                "complete": True,
                "corpus_sha256": before_descriptor.corpus_sha256,
                "corpus_rows": corpus_rows,
                "before": _run_manifest_value(before_descriptor),
                "after": _run_manifest_value(after_descriptor),
                "relations": relation_manifest,
                "summary": {
                    "filename": temporary_summary.name,
                    "sha256": _sha256_file(temporary_summary),
                },
            },
        )
    relation_paths = {
        name: destination / f"{name}.parquet" for name in _RELATION_SCHEMAS
    }
    return PreprocessingComparisonArtifacts(
        output_dir=destination,
        summary_path=destination / SUMMARY_FILENAME,
        manifest_path=destination / MANIFEST_FILENAME,
        relation_paths=relation_paths,
    )


@contextmanager
def _load_descriptor(
    *,
    label: str,
    dataset_id: str,
    corpus: Path,
    preprocessing: Path | str,
    evaluation: Path | str | None,
) -> Iterator[RunDescriptor]:
    try:
        with admitted_run_descriptor(
            label=label,
            dataset_id=dataset_id,
            corpus_path=corpus,
            preprocessing=preprocessing,
            candidate_evaluation=evaluation,
        ) as descriptor:
            yield descriptor
    except RunValidationError as exc:
        raise PreprocessingComparisonError(
            f"{label} immutable bundle is invalid: {exc}"
        ) from exc


@contextmanager
def _comparison_store(root: Path) -> Iterator[sqlite3.Connection]:
    with tempfile.TemporaryDirectory(
        prefix="comparison-store-", dir=root
    ) as store_root:
        connection = sqlite3.connect(Path(store_root) / "comparison.sqlite3")
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute(
                "CREATE TABLE corpus(sample_id TEXT PRIMARY KEY)"
            )
            for run in ("before", "after"):
                connection.executescript(
                    f"""
                    CREATE TABLE {run}_results(
                        sample_id TEXT PRIMARY KEY REFERENCES corpus(sample_id),
                        raw_output_sha256 TEXT,
                        decoder_output_presence TEXT NOT NULL,
                        outcome TEXT NOT NULL,
                        outcome_code TEXT,
                        failure_code TEXT,
                        failed_step TEXT,
                        cause TEXT,
                        propagated_through_json TEXT,
                        final_candidate_count INTEGER NOT NULL
                    );
                    CREATE TABLE {run}_candidates(
                        sample_id TEXT NOT NULL REFERENCES corpus(sample_id),
                        candidate_id TEXT NOT NULL,
                        candidate_index INTEGER NOT NULL,
                        source_sha256 TEXT NOT NULL,
                        PRIMARY KEY(sample_id, candidate_id)
                    );
                    CREATE INDEX {run}_candidate_order
                    ON {run}_candidates(sample_id, candidate_index);
                    CREATE TABLE {run}_provenance(
                        sample_id TEXT NOT NULL,
                        candidate_id TEXT NOT NULL,
                        path_json TEXT NOT NULL,
                        path_count INTEGER NOT NULL,
                        PRIMARY KEY(sample_id, candidate_id, path_json),
                        FOREIGN KEY(sample_id, candidate_id)
                            REFERENCES {run}_candidates(sample_id, candidate_id)
                    );
                    CREATE TABLE {run}_memberships(
                        sample_id TEXT NOT NULL,
                        candidate_id TEXT NOT NULL,
                        evaluation_key TEXT NOT NULL,
                        membership_json TEXT NOT NULL,
                        PRIMARY KEY(sample_id, candidate_id)
                    );
                    CREATE TABLE {run}_evaluation_results(
                        evaluation_key TEXT PRIMARY KEY,
                        result_json TEXT NOT NULL
                    );
                    """
                )
            yield connection
        finally:
            connection.close()


def _load_corpus(connection: sqlite3.Connection, path: Path) -> int:
    parquet = pq.ParquetFile(path)
    if "sample_id" not in parquet.schema_arrow.names:
        raise PreprocessingComparisonError(
            "comparison corpus is missing sample_id"
        )
    count = 0
    for row in _iter_parquet_rows(path, ("sample_id",)):
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise PreprocessingComparisonError(
                "comparison corpus contains an invalid sample_id"
            )
        try:
            connection.execute(
                "INSERT INTO corpus(sample_id) VALUES (?)", (sample_id,)
            )
        except sqlite3.IntegrityError as exc:
            raise PreprocessingComparisonError(
                f"comparison corpus contains duplicate sample_id: {sample_id}"
            ) from exc
        count += 1
    connection.commit()
    return count


def _load_run(
    connection: sqlite3.Connection,
    run: str,
    descriptor: RunDescriptor,
    corpus_rows: int,
) -> None:
    _load_results(connection, run, descriptor)
    _load_candidates(connection, run, descriptor)
    if descriptor.has_evaluation:
        _load_evaluation(connection, run, descriptor)
    result_rows = cast(
        int,
        connection.execute(f"SELECT count(*) FROM {run}_results").fetchone()[
            0
        ],
    )
    if result_rows != corpus_rows:
        raise PreprocessingComparisonError(
            f"{descriptor.label} results sample identities do not match corpus"
        )
    _validate_candidate_counts(connection, run, descriptor.label)
    connection.commit()


def _load_results(
    connection: sqlite3.Connection, run: str, descriptor: RunDescriptor
) -> None:
    columns = (
        "sample_id",
        "raw_output_sha256",
        "decoder_output_presence",
        "outcome",
        "outcome_code",
        "failure_code",
        "failed_step",
        "cause",
        "propagated_through",
        "final_candidate_count",
    )
    sql = f"INSERT INTO {run}_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    for row in _iter_parquet_rows(descriptor.results_path, columns):
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise PreprocessingComparisonError(
                "preprocessing result contains invalid sample_id"
            )
        propagated = row["propagated_through"]
        try:
            connection.execute(
                sql,
                (
                    sample_id,
                    row["raw_output_sha256"],
                    row["decoder_output_presence"],
                    row["outcome"],
                    row["outcome_code"],
                    row["failure_code"],
                    row["failed_step"],
                    row["cause"],
                    (
                        _canonical_json(propagated)
                        if propagated is not None
                        else None
                    ),
                    row["final_candidate_count"],
                ),
            )
        except sqlite3.IntegrityError as exc:
            if _corpus_contains(connection, sample_id):
                message = (
                    f"duplicate preprocessing result identity: {(sample_id,)}"
                )
            else:
                message = (
                    f"{descriptor.label} results sample identities "
                    "do not match corpus"
                )
            raise PreprocessingComparisonError(message) from exc


def _load_candidates(
    connection: sqlite3.Connection, run: str, descriptor: RunDescriptor
) -> None:
    columns = (
        "sample_id",
        "candidate_id",
        "candidate_index",
        "cleaned_source",
        "source_sha256",
        "origins",
    )
    for row in _iter_parquet_rows(descriptor.candidates_path, columns):
        sample_id = row["sample_id"]
        candidate_id = row["candidate_id"]
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or not isinstance(candidate_id, str)
            or not candidate_id
        ):
            raise PreprocessingComparisonError(
                f"{descriptor.label} candidate has invalid identity"
            )
        index = row["candidate_index"]
        source = row["cleaned_source"]
        source_sha256 = row["source_sha256"]
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise PreprocessingComparisonError(
                f"{descriptor.label} candidate has invalid index: "
                f"{sample_id}/{candidate_id}"
            )
        if (
            not isinstance(source, str)
            or not isinstance(source_sha256, str)
            or hashlib.sha256(source.encode("utf-8")).hexdigest()
            != source_sha256
        ):
            raise PreprocessingComparisonError(
                f"{descriptor.label} candidate source identity mismatch: "
                f"{sample_id}/{candidate_id}"
            )
        try:
            connection.execute(
                f"INSERT INTO {run}_candidates VALUES (?, ?, ?, ?)",
                (sample_id, candidate_id, index, source_sha256),
            )
        except sqlite3.IntegrityError as exc:
            if _corpus_contains(connection, sample_id):
                message = (
                    "duplicate preprocessing candidate identity: "
                    f"{(sample_id, candidate_id)}"
                )
            else:
                message = f"{descriptor.label} candidate has invalid identity"
            raise PreprocessingComparisonError(message) from exc
        for origin in normalize_origins(row["origins"]):
            connection.execute(
                f"""
                INSERT INTO {run}_provenance
                    (sample_id, candidate_id, path_json, path_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(sample_id, candidate_id, path_json)
                DO UPDATE SET path_count = path_count + 1
                """,
                (sample_id, candidate_id, _canonical_json(origin)),
            )


def _load_evaluation(
    connection: sqlite3.Connection, run: str, descriptor: RunDescriptor
) -> None:
    assert descriptor.candidate_membership_path is not None
    assert descriptor.candidate_results_path is not None
    for row in _iter_parquet_rows(descriptor.candidate_membership_path):
        sample_id = row.get("sample_id")
        candidate_id = row.get("candidate_id")
        evaluation_key = row.get("evaluation_key")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or not isinstance(candidate_id, str)
            or not candidate_id
            or not isinstance(evaluation_key, str)
            or not evaluation_key
        ):
            raise PreprocessingComparisonError(
                "evaluation membership contains invalid identity"
            )
        try:
            connection.execute(
                f"INSERT INTO {run}_memberships VALUES (?, ?, ?, ?)",
                (
                    sample_id,
                    candidate_id,
                    evaluation_key,
                    _canonical_json(row),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise PreprocessingComparisonError(
                "duplicate evaluation membership identity: "
                f"{(sample_id, candidate_id)}"
            ) from exc
    for row in _iter_parquet_rows(descriptor.candidate_results_path):
        evaluation_key = row.get("evaluation_key")
        if not isinstance(evaluation_key, str) or not evaluation_key:
            raise PreprocessingComparisonError(
                "evaluation result contains invalid evaluation_key"
            )
        compact = {
            key: value for key, value in row.items() if key != "cleaned_source"
        }
        try:
            connection.execute(
                f"INSERT INTO {run}_evaluation_results VALUES (?, ?)",
                (evaluation_key, _canonical_json(compact)),
            )
        except sqlite3.IntegrityError as exc:
            raise PreprocessingComparisonError(
                f"duplicate evaluation result identity: {(evaluation_key,)}"
            ) from exc


def _validate_candidate_counts(
    connection: sqlite3.Connection, run: str, label: str
) -> None:
    rows = connection.execute(
        f"""
        SELECT
            results.sample_id,
            results.final_candidate_count AS expected_count,
            count(candidates.candidate_id) AS actual_count,
            count(DISTINCT candidates.candidate_index) AS distinct_indices,
            min(candidates.candidate_index) AS minimum_index,
            max(candidates.candidate_index) AS maximum_index
        FROM {run}_results AS results
        LEFT JOIN {run}_candidates AS candidates USING(sample_id)
        GROUP BY results.sample_id, results.final_candidate_count
        ORDER BY results.sample_id
        """
    )
    for row in rows:
        expected = cast(int, row["expected_count"])
        actual = cast(int, row["actual_count"])
        if expected != actual:
            raise PreprocessingComparisonError(
                f"{label} final candidate count mismatch: {row['sample_id']}"
            )
        contiguous = actual == 0 or (
            row["minimum_index"] == 0
            and row["maximum_index"] == actual - 1
            and row["distinct_indices"] == actual
        )
        if not contiguous:
            raise PreprocessingComparisonError(
                f"{label} candidate indices are not contiguous: "
                f"{row['sample_id']}"
            )


def _corpus_contains(connection: sqlite3.Connection, sample_id: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM corpus WHERE sample_id = ?", (sample_id,)
        ).fetchone()
        is not None
    )


def _iter_parquet_rows(
    path: Path, columns: tuple[str, ...] | None = None
) -> Iterator[dict[str, object]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=_INPUT_BATCH_SIZE,
        columns=list(columns) if columns is not None else None,
    ):
        names = tuple(batch.schema.names)
        values = [batch.column(name).to_pylist() for name in names]
        for items in zip(*values, strict=True):
            yield dict(zip(names, items, strict=True))


def _comparison_rows(
    connection: sqlite3.Connection, relation: str
) -> Iterator[dict[str, object]]:
    factories: Mapping[
        str,
        Callable[[sqlite3.Connection], Iterator[dict[str, object]]],
    ] = {
        "sample_outcome_transitions": _sample_transition_rows,
        "candidate_changes": _candidate_change_rows,
        "provenance_path_deltas": _provenance_delta_rows,
        "evaluation_membership_changes": _evaluation_membership_rows,
        "evaluation_result_changes": _evaluation_result_rows,
    }
    yield from factories[relation](connection)


def _sample_transition_rows(
    connection: sqlite3.Connection,
) -> Iterator[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT
            before.sample_id,
            before.raw_output_sha256 AS before_raw_output_sha256,
            after.raw_output_sha256 AS after_raw_output_sha256,
            before.decoder_output_presence AS before_decoder_output_presence,
            after.decoder_output_presence AS after_decoder_output_presence,
            before.outcome AS before_outcome,
            after.outcome AS after_outcome,
            before.outcome_code AS before_outcome_code,
            after.outcome_code AS after_outcome_code,
            before.failure_code AS before_failure_code,
            after.failure_code AS after_failure_code,
            before.failed_step AS before_failed_step,
            after.failed_step AS after_failed_step,
            before.cause AS before_cause,
            after.cause AS after_cause,
            before.propagated_through_json
                AS before_propagated_through_json,
            after.propagated_through_json AS after_propagated_through_json,
            before.final_candidate_count AS before_final_candidate_count,
            after.final_candidate_count AS after_final_candidate_count
        FROM before_results AS before
        JOIN after_results AS after USING(sample_id)
        ORDER BY before.sample_id
        """
    )
    for row in rows:
        old = {
            field: _result_field(row, "before", field)
            for field in _SEMANTIC_RESULT_FIELDS
        }
        new = {
            field: _result_field(row, "after", field)
            for field in _SEMANTIC_RESULT_FIELDS
        }
        output_changed = (
            old["raw_output_sha256"] != new["raw_output_sha256"]
            or old["decoder_output_presence"] != new["decoder_output_presence"]
        )
        outcome_changed = old["outcome"] != new["outcome"]
        changed_fields = [
            field
            for field in _SEMANTIC_RESULT_FIELDS
            if old[field] != new[field]
        ]
        semantic_result_changed = bool(changed_fields)
        yield {
            "sample_id": row["sample_id"],
            "before_output_sha256": old["raw_output_sha256"],
            "after_output_sha256": new["raw_output_sha256"],
            "before_output_presence": old["decoder_output_presence"],
            "after_output_presence": new["decoder_output_presence"],
            "before_outcome": old["outcome"],
            "after_outcome": new["outcome"],
            "before_outcome_code": old["outcome_code"],
            "after_outcome_code": new["outcome_code"],
            "before_failure_code": old["failure_code"],
            "after_failure_code": new["failure_code"],
            "before_failed_step": old["failed_step"],
            "after_failed_step": new["failed_step"],
            "before_cause": old["cause"],
            "after_cause": new["cause"],
            "before_propagated_through": old["propagated_through"],
            "after_propagated_through": new["propagated_through"],
            "before_final_candidate_count": old["final_candidate_count"],
            "after_final_candidate_count": new["final_candidate_count"],
            "output_identity_changed": output_changed,
            "outcome_changed": outcome_changed,
            "semantic_result_changed": semantic_result_changed,
            "changed_fields": changed_fields,
            "change": (
                "semantic_result_changed"
                if semantic_result_changed
                else "unchanged"
            ),
        }


def _result_field(row: sqlite3.Row, prefix: str, field: str) -> object:
    if field == "propagated_through":
        value = row[f"{prefix}_propagated_through_json"]
        return json.loads(value) if value is not None else None
    return row[f"{prefix}_{field}"]


def _candidate_change_rows(
    connection: sqlite3.Connection,
) -> Iterator[dict[str, object]]:
    rows = connection.execute(
        """
        WITH identities AS (
            SELECT sample_id, candidate_id FROM before_candidates
            UNION
            SELECT sample_id, candidate_id FROM after_candidates
        )
        SELECT
            identities.sample_id,
            identities.candidate_id,
            before.candidate_id IS NOT NULL AS before_present,
            after.candidate_id IS NOT NULL AS after_present,
            before.candidate_index AS before_candidate_index,
            after.candidate_index AS after_candidate_index,
            before.source_sha256 AS before_source_sha256,
            after.source_sha256 AS after_source_sha256
        FROM identities
        LEFT JOIN before_candidates AS before USING(sample_id, candidate_id)
        LEFT JOIN after_candidates AS after USING(sample_id, candidate_id)
        ORDER BY identities.sample_id, identities.candidate_id
        """
    )
    for row in rows:
        before_present = bool(row["before_present"])
        after_present = bool(row["after_present"])
        membership_changed = before_present != after_present or (
            before_present
            and after_present
            and row["before_candidate_index"] != row["after_candidate_index"]
        )
        source_changed = (
            before_present
            and after_present
            and row["before_source_sha256"] != row["after_source_sha256"]
        )
        if not before_present:
            change = "added"
        elif not after_present:
            change = "removed"
        elif membership_changed or source_changed:
            change = "modified"
        else:
            change = "unchanged"
        yield {
            "sample_id": row["sample_id"],
            "candidate_id": row["candidate_id"],
            "before_present": before_present,
            "after_present": after_present,
            "before_candidate_index": row["before_candidate_index"],
            "after_candidate_index": row["after_candidate_index"],
            "before_source_sha256": row["before_source_sha256"],
            "after_source_sha256": row["after_source_sha256"],
            "membership_changed": membership_changed,
            "source_changed": source_changed,
            "change": change,
        }


def _provenance_delta_rows(
    connection: sqlite3.Connection,
) -> Iterator[dict[str, object]]:
    rows = connection.execute(
        """
        WITH identities AS (
            SELECT sample_id, candidate_id, path_json FROM before_provenance
            UNION
            SELECT sample_id, candidate_id, path_json FROM after_provenance
        )
        SELECT
            identities.sample_id,
            identities.candidate_id,
            identities.path_json,
            coalesce(before.path_count, 0) AS before_count,
            coalesce(after.path_count, 0) AS after_count
        FROM identities
        LEFT JOIN before_provenance AS before
            USING(sample_id, candidate_id, path_json)
        LEFT JOIN after_provenance AS after
            USING(sample_id, candidate_id, path_json)
        ORDER BY
            identities.sample_id,
            identities.candidate_id,
            identities.path_json
        """
    )
    for row in rows:
        old = cast(int, row["before_count"])
        new = cast(int, row["after_count"])
        change = "unchanged"
        if old == 0:
            change = "added"
        elif new == 0:
            change = "removed"
        elif old != new:
            change = "count_changed"
        yield {
            "sample_id": row["sample_id"],
            "candidate_id": row["candidate_id"],
            "path_json": row["path_json"],
            "before_count": old,
            "after_count": new,
            "count_delta": new - old,
            "change": change,
        }


def _evaluation_membership_rows(
    connection: sqlite3.Connection,
) -> Iterator[dict[str, object]]:
    rows = connection.execute(
        """
        WITH identities AS (
            SELECT sample_id, candidate_id FROM before_memberships
            UNION
            SELECT sample_id, candidate_id FROM after_memberships
        )
        SELECT
            identities.sample_id,
            identities.candidate_id,
            before.evaluation_key AS before_evaluation_key,
            after.evaluation_key AS after_evaluation_key,
            before.membership_json AS before_membership_json,
            after.membership_json AS after_membership_json
        FROM identities
        LEFT JOIN before_memberships AS before USING(sample_id, candidate_id)
        LEFT JOIN after_memberships AS after USING(sample_id, candidate_id)
        ORDER BY identities.sample_id, identities.candidate_id
        """
    )
    for row in rows:
        old = row["before_membership_json"]
        new = row["after_membership_json"]
        yield {
            "sample_id": row["sample_id"],
            "candidate_id": row["candidate_id"],
            "before_present": old is not None,
            "after_present": new is not None,
            "before_evaluation_key": row["before_evaluation_key"],
            "after_evaluation_key": row["after_evaluation_key"],
            "before_membership_json": old,
            "after_membership_json": new,
            "change": _optional_json_change(old, new),
        }


def _evaluation_result_rows(
    connection: sqlite3.Connection,
) -> Iterator[dict[str, object]]:
    rows = connection.execute(
        """
        WITH identities AS (
            SELECT sample_id, candidate_id FROM before_memberships
            UNION
            SELECT sample_id, candidate_id FROM after_memberships
        )
        SELECT
            identities.sample_id,
            identities.candidate_id,
            before_membership.evaluation_key AS before_evaluation_key,
            after_membership.evaluation_key AS after_evaluation_key,
            before_result.result_json AS before_result_json,
            after_result.result_json AS after_result_json
        FROM identities
        LEFT JOIN before_memberships AS before_membership
            USING(sample_id, candidate_id)
        LEFT JOIN after_memberships AS after_membership
            USING(sample_id, candidate_id)
        LEFT JOIN before_evaluation_results AS before_result
            ON before_result.evaluation_key =
                before_membership.evaluation_key
        LEFT JOIN after_evaluation_results AS after_result
            ON after_result.evaluation_key =
                after_membership.evaluation_key
        ORDER BY identities.sample_id, identities.candidate_id
        """
    )
    for row in rows:
        old = row["before_result_json"]
        new = row["after_result_json"]
        yield {
            "sample_id": row["sample_id"],
            "candidate_id": row["candidate_id"],
            "before_evaluation_key": row["before_evaluation_key"],
            "after_evaluation_key": row["after_evaluation_key"],
            "before_result_json": old,
            "after_result_json": new,
            "change": _optional_json_change(old, new),
        }


def _optional_json_change(
    before: str | None,
    after: str | None,
) -> str:
    if before is None:
        return "added"
    if after is None:
        return "removed"
    return "unchanged" if before == after else "modified"


def _evaluation_coordinate_summary(
    before: RunDescriptor, after: RunDescriptor
) -> dict[str, object] | None:
    if not before.has_evaluation:
        return None
    old = _semantic_coordinates(before)
    new = _semantic_coordinates(after)
    fields = sorted(set(old) | set(new))
    return {
        "before": old,
        "after": new,
        "changed_fields": [
            field for field in fields if old.get(field) != new.get(field)
        ],
    }


def _semantic_coordinates(descriptor: RunDescriptor) -> Mapping[str, object]:
    assert descriptor.evaluation_coordinates is not None
    return {
        key: thaw_json(value)
        for key, value in sorted(descriptor.evaluation_coordinates.items())
        if key
        not in {
            "corpus_sha256",
            "preprocessing_run",
            "evaluation_identity",
        }
    }


def _run_manifest_value(descriptor: RunDescriptor) -> dict[str, object]:
    return {
        "run_id": descriptor.run_id,
        "dataset_id": descriptor.dataset_id,
        "preprocessing_schema_version": descriptor.preprocessing_schema_version,
        "preprocessing_manifest_sha256": (
            descriptor.preprocessing_manifest_sha256
        ),
        "artifact_sha256": dict(sorted(descriptor.artifact_sha256.items())),
        "evaluation_manifest_sha256": descriptor.evaluation_manifest_sha256,
    }


def _write_relation(
    path: Path,
    rows: Iterator[dict[str, object]],
    schema: pa.Schema,
    consume: Callable[[Mapping[str, object]], None],
) -> int:
    row_count = 0
    try:
        with pq.ParquetWriter(
            path,
            schema,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            version="2.6",
        ) as writer:
            batch: list[dict[str, object]] = []
            for row in rows:
                consume(row)
                batch.append(row)
                row_count += 1
                if len(batch) == _OUTPUT_BATCH_SIZE:
                    writer.write_table(
                        pa.Table.from_pylist(batch, schema=schema)
                    )
                    batch.clear()
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return row_count


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = (
    "COMPARISON_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "SUMMARY_FILENAME",
    "PreprocessingComparisonArtifacts",
    "PreprocessingComparisonError",
    "compare_preprocessing_runs",
)
