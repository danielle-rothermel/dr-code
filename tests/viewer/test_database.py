from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import dr_code.corpus.stable_files as stable_files_module
from dr_code.execution import run_subprocess
import dr_code.viewer.database as viewer_database
from dr_code.viewer.database import (
    DatabaseOwnershipError,
    DatabaseSchemaError,
    ViewerDatabase,
    database_owner_lock_path,
)
import dr_code.viewer.database as database_module
from dr_code.viewer.domain import (
    InvalidQueryError,
    InvalidTaskAnnotationError,
    RunDescriptor,
    TaskAnnotationOrigin,
    TaskAnnotationPublicationIntent,
    TaskAnnotationProvenance,
    TaskAnnotation,
    TaskIdentity,
    Verdict,
    validate_task_annotation,
)
from viewer.helpers import write_bundle

CORPUS = "a" * 64
OUTPUT = "b" * 64
EXPERIMENT = "c" * 64


@dataclass(frozen=True, slots=True)
class _DatabaseSnapshot:
    contents: bytes
    catalog_objects: frozenset[viewer_database._CatalogObject]
    table_signatures: tuple[
        tuple[
            str,
            tuple[
                tuple[tuple[object, ...], ...],
                tuple[tuple[object, ...], ...],
            ],
        ],
        ...,
    ]
    version: tuple[tuple[object, ...], ...] | None


def _create_unversioned_schema(path: Path) -> None:
    with duckdb.connect(str(path)) as connection:
        for statement in viewer_database._LEGACY_SCHEMA:
            connection.execute(statement)


def _create_persisted_schema(
    path: Path,
    statements: tuple[str, ...],
    schema_version: int | None,
) -> None:
    with duckdb.connect(str(path)) as connection:
        for statement in statements:
            connection.execute(statement)
        if schema_version is not None:
            connection.execute(viewer_database._VIEWER_SCHEMA)
            connection.execute(
                "INSERT INTO viewer_schema VALUES (?)", [schema_version]
            )


def _database_snapshot(
    path: Path,
) -> _DatabaseSnapshot:
    contents = path.read_bytes()
    with duckdb.connect(str(path), read_only=True) as connection:
        catalog_objects = viewer_database._user_catalog_objects(connection)
        tables = tuple(sorted(viewer_database._main_tables(connection)))
        signatures = tuple(
            (
                table,
                viewer_database._table_schema_signature(connection, table),
            )
            for table in tables
        )
        version = (
            tuple(
                connection.execute(
                    "SELECT schema_version FROM viewer_schema"
                ).fetchall()
            )
            if "viewer_schema" in tables
            else None
        )
    return _DatabaseSnapshot(contents, catalog_objects, signatures, version)


def _table_content_signatures(
    connection: duckdb.DuckDBPyConnection,
    tables: frozenset[str],
) -> dict[str, tuple[int, int | None]]:
    signatures = {}
    for table in tables:
        columns = [
            row[1]
            for row in connection.execute(
                f"PRAGMA table_info('{table}')"
            ).fetchall()
        ]
        hashed_columns = ", ".join(f'"{column}"' for column in columns)
        row = connection.execute(
            f"SELECT count(*), bit_xor(hash({hashed_columns})) FROM {table}"
        ).fetchone()
        assert row is not None
        signatures[table] = cast(tuple[int, int | None], row)
    return signatures


def _table_counts(
    connection: duckdb.DuckDBPyConnection,
    tables: tuple[str, ...],
) -> tuple[int, ...]:
    counts = []
    for table in tables:
        row = connection.execute(f"SELECT count(*) FROM {table}").fetchone()
        assert row is not None
        counts.append(cast(int, row[0]))
    return tuple(counts)


def test_database_owner_lock_path_is_canonical(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / ".." / "viewer.duckdb"

    assert (
        database_owner_lock_path(database_path)
        == (tmp_path / ".viewer.duckdb.owner.lock").resolve()
    )
    with pytest.raises(ValueError, match="no ownership lock"):
        database_owner_lock_path(":memory:")


def test_in_memory_classification_mutexes_are_independent() -> None:
    with (
        ViewerDatabase(":memory:") as first,
        ViewerDatabase(":memory:") as second,
    ):
        entered = threading.Event()

        def enter_second() -> None:
            with second.classification_serialization():
                entered.set()

        with first.classification_serialization():
            thread = threading.Thread(target=enter_second)
            thread.start()
            assert entered.wait(timeout=5)
        thread.join(timeout=5)


def test_closed_database_rejects_classification_serialization() -> None:
    database = ViewerDatabase(":memory:")
    database.close()

    with pytest.raises(DatabaseOwnershipError, match="connection is closed"):
        with database.classification_serialization():
            raise AssertionError("closed database entered serialization")


def _task_identity(task_id: str) -> str:
    return hashlib.sha256(task_id.encode()).hexdigest()


def _with_membership(
    descriptor: RunDescriptor,
    path: Path,
    task_ids: list[str],
) -> RunDescriptor:
    pq.write_table(
        pa.table(
            {
                "task_id": task_ids,
                "task_identity": [
                    _task_identity(task_id) for task_id in task_ids
                ],
            }
        ),
        path,
        row_group_size=127,
    )
    return replace(
        descriptor,
        candidate_membership_path=path,
        artifact_sha256={
            **descriptor.artifact_sha256,
            "candidate_membership": hashlib.sha256(
                path.read_bytes()
            ).hexdigest(),
        },
    )


def _create_existing_database(path: Path) -> None:
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            CREATE TABLE runs (
                run_id VARCHAR PRIMARY KEY,
                label VARCHAR NOT NULL,
                descriptor_json VARCHAR NOT NULL,
                manifest_sha256 VARCHAR NOT NULL,
                corpus_sha256 VARCHAR NOT NULL,
                definition_id VARCHAR NOT NULL,
                registered_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
            );
            CREATE TABLE annotations (
                corpus_sha256 VARCHAR NOT NULL,
                sample_id VARCHAR NOT NULL,
                decoder_output_sha256 VARCHAR NOT NULL,
                verdict VARCHAR CHECK (
                    verdict IN ('should_be_parseable', 'expected_no_code')
                ),
                note VARCHAR,
                created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
                PRIMARY KEY (
                    corpus_sha256, sample_id, decoder_output_sha256
                )
            );
            CREATE TABLE tags (
                tag_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                normalized_name VARCHAR NOT NULL UNIQUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
            );
            CREATE TABLE annotation_tags (
                corpus_sha256 VARCHAR NOT NULL,
                sample_id VARCHAR NOT NULL,
                decoder_output_sha256 VARCHAR NOT NULL,
                tag_id VARCHAR NOT NULL REFERENCES tags(tag_id),
                PRIMARY KEY (
                    corpus_sha256, sample_id, decoder_output_sha256, tag_id
                )
            )
            """
        )


def _create_legacy_task_tables(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    connection.execute(
        """
        CREATE TABLE registered_tasks (
            run_id VARCHAR NOT NULL,
            dataset_id VARCHAR NOT NULL,
            task_id VARCHAR NOT NULL,
            PRIMARY KEY (run_id, dataset_id, task_id)
        );
        CREATE TABLE task_annotations (
            dataset_id VARCHAR NOT NULL,
            task_id VARCHAR NOT NULL,
            origin VARCHAR NOT NULL,
            PRIMARY KEY (dataset_id, task_id)
        );
        CREATE TABLE task_annotation_tags (
            dataset_id VARCHAR NOT NULL,
            task_id VARCHAR NOT NULL,
            tag_id VARCHAR NOT NULL,
            PRIMARY KEY (dataset_id, task_id, tag_id)
        )
        """
    )


def _machine_candidate(
    dataset_id: str,
    task_id: str,
    *,
    category: str,
) -> TaskAnnotation:
    return validate_task_annotation(
        identity=TaskIdentity(
            dataset_id=dataset_id,
            task_id=task_id,
            task_identity=_task_identity(task_id),
        ),
        origin=TaskAnnotationOrigin.MACHINE,
        category=category,
        note=None,
        tags=(),
        provenance=TaskAnnotationProvenance(
            model="model",
            taxonomy_version="taxonomy",
            repeats=1,
            agreement=1,
            extra={
                "producer": "classifier",
                "experiment_identity": EXPERIMENT,
            },
        ),
    )


def _publication_intent(
    tmp_path: Path,
    *,
    output_name: str = "details.jsonl",
) -> TaskAnnotationPublicationIntent:
    output = (tmp_path / output_name).resolve()
    staged = output.parent / f".{output.name}.publication"
    return TaskAnnotationPublicationIntent(
        producer="classifier",
        experiment_identity=EXPERIMENT,
        output_path=str(output),
        staged_path=str(staged),
        prior_sha256=None,
        intended_sha256="e" * 64,
    )


def test_machine_batch_rolls_back_all_rows_on_late_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    with ViewerDatabase(":memory:") as database:
        candidate = _machine_candidate(
            "dataset",
            "Task/1",
            category="new",
        )

        def fail_tags(*args, **kwargs):
            raise RuntimeError("injected transaction failure")

        monkeypatch.setattr(
            database,
            "_replace_task_annotation_tags",
            fail_tags,
        )
        intent = _publication_intent(tmp_path)
        database.begin_task_annotation_publication(intent)
        with pytest.raises(RuntimeError, match="injected"):
            database.finalize_task_annotation_publication(
                (candidate.identity,),
                (candidate,),
                intent=intent,
            )

        assert (
            database.get_task_annotation(
                "dataset", "Task/1", _task_identity("Task/1")
            )
            is None
        )
        assert (
            database.get_task_annotation_publication_intent(intent.output_path)
            == intent
        )


def test_machine_batch_requires_the_exact_pending_intent(
    tmp_path: Path,
) -> None:
    with ViewerDatabase(":memory:") as database:
        candidate = _machine_candidate(
            "dataset",
            "Task/1",
            category="new",
        )
        intent = _publication_intent(tmp_path)
        database.begin_task_annotation_publication(intent)
        wrong = TaskAnnotationPublicationIntent(
            producer=intent.producer,
            experiment_identity=intent.experiment_identity,
            output_path=intent.output_path,
            staged_path=intent.staged_path,
            prior_sha256=intent.prior_sha256,
            intended_sha256="f" * 64,
        )

        with pytest.raises(
            InvalidTaskAnnotationError,
            match="exact task annotation publication intent",
        ):
            database.finalize_task_annotation_publication(
                (candidate.identity,),
                (candidate,),
                intent=wrong,
            )

        assert (
            database.get_task_annotation(
                "dataset", "Task/1", _task_identity("Task/1")
            )
            is None
        )
        assert (
            database.get_task_annotation_publication_intent(intent.output_path)
            == intent
        )


def test_machine_batch_deletes_only_owned_rows_then_their_tags(
    tmp_path: Path,
) -> None:
    with ViewerDatabase(":memory:") as database:
        owned_tag = database.create_tag("owned")
        other_tag = database.create_tag("other")
        owned = database.put_machine_task_annotation(
            "dataset",
            "Task/1",
            _task_identity("Task/1"),
            category="owned",
            tag_ids=(owned_tag.tag_id,),
            provenance=TaskAnnotationProvenance(
                model="model",
                taxonomy_version="taxonomy",
                repeats=1,
                agreement=1,
                extra={
                    "producer": "classifier",
                    "experiment_identity": EXPERIMENT,
                },
            ),
        )
        other = database.put_machine_task_annotation(
            "dataset",
            "Task/2",
            _task_identity("Task/2"),
            category="other",
            tag_ids=(other_tag.tag_id,),
            provenance=TaskAnnotationProvenance(
                model="model",
                taxonomy_version="taxonomy",
                repeats=1,
                agreement=1,
                extra={
                    "producer": "classifier",
                    "experiment_identity": "d" * 64,
                },
            ),
        )
        assert owned.annotation.tags == (owned_tag,)
        assert other.annotation.tags == (other_tag,)

        intent = _publication_intent(tmp_path)
        database.begin_task_annotation_publication(intent)
        result = database.finalize_task_annotation_publication(
            (
                TaskIdentity("dataset", "Task/1", _task_identity("Task/1")),
                TaskIdentity("dataset", "Task/2", _task_identity("Task/2")),
            ),
            (),
            intent=intent,
        )

        assert result.removed == 1
        assert (
            database.get_task_annotation(
                "dataset", "Task/1", _task_identity("Task/1")
            )
            is None
        )
        assert database.get_task_annotation(
            "dataset", "Task/2", _task_identity("Task/2")
        ) == (other.annotation)
        tag_links = database.connection.execute(
            "SELECT task_id, tag_id FROM task_annotation_tags ORDER BY task_id"
        ).fetchall()
        assert tag_links == [("Task/2", other_tag.tag_id)]
        assert (
            database.get_task_annotation_publication_intent(intent.output_path)
            is None
        )


def test_machine_publication_streams_many_tasks_in_bounded_batches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    with ViewerDatabase(":memory:") as database:
        identities = tuple(
            TaskIdentity(
                "dataset",
                f"Task/{index:03d}",
                _task_identity(f"Task/{index:03d}"),
            )
            for index in range(10)
        )
        candidates = tuple(
            _machine_candidate(
                identity.dataset_id,
                identity.task_id,
                category="batch",
            )
            for identity in identities
        )

        class NoLengthHintIterator:
            def __init__(self, values) -> None:
                self._values = iter(values)
                self.yielded = 0

            def __iter__(self):
                return self

            def __next__(self):
                value = next(self._values)
                self.yielded += 1
                return value

            def __length_hint__(self) -> int:
                raise AssertionError("publication input was materialized")

        scope = NoLengthHintIterator(identities)
        annotations = NoLengthHintIterator(candidates)

        monkeypatch.setattr(
            database_module,
            "_PUBLICATION_BATCH_SIZE",
            3,
        )
        original_batches = database._publication_stage_batches
        batch_sizes: list[int] = []

        def observed_batches(stage: str):
            for batch in original_batches(stage):
                batch_sizes.append(len(batch))
                yield batch

        monkeypatch.setattr(
            database,
            "_publication_stage_batches",
            observed_batches,
        )
        intent = _publication_intent(tmp_path)
        database.begin_task_annotation_publication(intent)

        result = database.finalize_task_annotation_publication(
            scope,
            annotations,
            intent=intent,
        )

        assert result.written == 10
        assert result.protected == 0
        assert result.removed == 0
        assert batch_sizes == [3, 3, 3, 1]
        assert scope.yielded == 10
        assert annotations.yielded == 10


def test_pending_intent_suppresses_matching_machine_but_not_human(
    tmp_path: Path,
) -> None:
    with ViewerDatabase(":memory:") as database:
        matching = database.put_machine_task_annotation(
            "dataset",
            "Task/1",
            _task_identity("Task/1"),
            category="matching",
            provenance=cast(
                TaskAnnotationProvenance,
                _machine_candidate(
                    "dataset", "Task/1", category="matching"
                ).provenance,
            ),
        )
        other = database.put_machine_task_annotation(
            "dataset",
            "Task/2",
            _task_identity("Task/2"),
            category="other",
            provenance=TaskAnnotationProvenance(
                model="model",
                taxonomy_version="taxonomy",
                repeats=1,
                agreement=1,
                extra={
                    "producer": "classifier",
                    "experiment_identity": "d" * 64,
                },
            ),
        )
        human = database.put_task_annotation(
            "dataset",
            "Task/3",
            _task_identity("Task/3"),
            category="human",
        )
        intent = _publication_intent(tmp_path)
        database.begin_task_annotation_publication(intent)

        assert (
            database.get_task_annotation(
                "dataset", "Task/1", _task_identity("Task/1")
            )
            is None
        )
        assert database.get_task_annotation(
            "dataset", "Task/2", _task_identity("Task/2")
        ) == (other.annotation)
        assert (
            database.get_task_annotation(
                "dataset", "Task/3", _task_identity("Task/3")
            )
            == human
        )
        listed = database._task_annotations()  # noqa: SLF001
        exported = database.export_task_annotations()

        assert matching.annotation not in listed
        assert listed == (other.annotation, human)
        assert [item["category"] for item in exported] == ["other", "human"]


def test_same_experiment_intents_are_keyed_by_path_and_suppress_until_all_clear(
    tmp_path: Path,
) -> None:
    with ViewerDatabase(":memory:") as database:
        machine = database.put_machine_task_annotation(
            "dataset",
            "Task/1",
            _task_identity("Task/1"),
            category="machine",
            provenance=cast(
                TaskAnnotationProvenance,
                _machine_candidate(
                    "dataset", "Task/1", category="machine"
                ).provenance,
            ),
        )
        first = _publication_intent(tmp_path, output_name="first.jsonl")
        second = _publication_intent(tmp_path, output_name="second.jsonl")

        database.begin_task_annotation_publication(first)
        database.begin_task_annotation_publication(second)
        database.abort_task_annotation_publication(first)
        assert (
            database.get_task_annotation(
                "dataset", "Task/1", _task_identity("Task/1")
            )
            is None
        )

        database.abort_task_annotation_publication(second)
        assert (
            database.get_task_annotation(
                "dataset", "Task/1", _task_identity("Task/1")
            )
            == machine.annotation
        )


def test_single_and_batch_machine_writes_share_protected_upsert_primitive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    with ViewerDatabase(":memory:") as database:
        original = database._upsert_task_annotation_row
        calls: list[TaskIdentity] = []

        def track(candidate):
            calls.append(candidate.identity)
            return original(candidate)

        monkeypatch.setattr(
            database,
            "_upsert_task_annotation_row",
            track,
        )
        database.put_machine_task_annotation(
            "dataset",
            "Task/1",
            _task_identity("Task/1"),
            category="single",
            provenance=cast(
                TaskAnnotationProvenance,
                _machine_candidate(
                    "dataset", "Task/1", category="single"
                ).provenance,
            ),
        )
        candidate = _machine_candidate(
            "dataset",
            "Task/2",
            category="batch",
        )
        intent = _publication_intent(tmp_path)
        database.begin_task_annotation_publication(intent)
        database.finalize_task_annotation_publication(
            (candidate.identity,),
            (candidate,),
            intent=intent,
        )

        assert calls == [
            TaskIdentity("dataset", "Task/1", _task_identity("Task/1")),
            TaskIdentity("dataset", "Task/2", _task_identity("Task/2")),
        ]


def test_existing_database_reopens_and_registers_runs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "viewer.duckdb"
    _create_existing_database(path)
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id="dataset/current",
        task_namespace="CurrentTask",
        with_evaluation=False,
    )

    with ViewerDatabase(path) as database:
        database.register_runs([descriptor])
        columns = [
            row[1]
            for row in database.connection.execute(
                "PRAGMA table_info('runs')"
            ).fetchall()
        ]
        stored = database.connection.execute(
            "SELECT descriptor_json FROM runs WHERE run_id = ?",
            [descriptor.run_id],
        ).fetchone()

    assert "dataset_id" not in columns
    assert stored is not None
    assert '"dataset_id":"dataset/current"' in stored[0]


def test_run_registration_uses_authenticated_membership_snapshot(
    tmp_path: Path,
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id="dataset/current",
        task_namespace="CurrentTask",
    )

    with ViewerDatabase(":memory:") as database:
        database.register_runs([descriptor])
        registrations = database.connection.execute(
            """
            SELECT dataset_id, task_id, task_identity
            FROM registered_tasks
            ORDER BY task_id
            """
        ).fetchall()

    assert registrations == [
        ("dataset/current", "CurrentTask/5", _task_identity("CurrentTask/5")),
        ("dataset/current", "CurrentTask/6", _task_identity("CurrentTask/6")),
    ]


def test_run_registration_rejects_membership_mutated_after_admission(
    tmp_path: Path,
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id="dataset/current",
        task_namespace="CurrentTask",
    )
    membership_path = descriptor.candidate_membership_path
    assert membership_path is not None

    with ViewerDatabase(":memory:") as database:
        database.register_runs([descriptor])
        before_runs = database.connection.execute(
            "SELECT * EXCLUDE (registered_at) FROM runs ORDER BY run_id"
        ).fetchall()
        before_tasks = database.connection.execute(
            "SELECT * FROM registered_tasks ORDER BY run_id, task_id"
        ).fetchall()

        membership_path.write_bytes(b"mutated after descriptor admission")
        with pytest.raises(
            InvalidQueryError,
            match="candidate membership does not match authenticated descriptor",
        ):
            database.register_runs([descriptor])

        assert (
            database.connection.execute(
                "SELECT * EXCLUDE (registered_at) FROM runs ORDER BY run_id"
            ).fetchall()
            == before_runs
        )
        assert (
            database.connection.execute(
                "SELECT * FROM registered_tasks ORDER BY run_id, task_id"
            ).fetchall()
            == before_tasks
        )


def test_run_registration_rejects_membership_hash_mismatch(
    tmp_path: Path,
) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    descriptor = replace(
        descriptor,
        artifact_sha256={
            **descriptor.artifact_sha256,
            "candidate_membership": "0" * 64,
        },
    )

    with ViewerDatabase(":memory:") as database:
        with pytest.raises(
            InvalidQueryError,
            match="candidate membership does not match authenticated descriptor",
        ):
            database.register_runs([descriptor])

        assert (
            database.connection.execute("SELECT * FROM runs").fetchall() == []
        )
        assert (
            database.connection.execute(
                "SELECT * FROM registered_tasks"
            ).fetchall()
            == []
        )


def test_run_registration_uses_captured_bytes_after_same_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id="dataset/current",
        task_namespace="CurrentTask",
    )
    membership_path = descriptor.candidate_membership_path
    assert membership_path is not None
    replacement_path = tmp_path / "replacement-membership.parquet"
    replacement_task_id = "Replacement/1"
    pq.write_table(
        pa.table(
            {
                "task_id": [replacement_task_id],
                "task_identity": [_task_identity(replacement_task_id)],
            }
        ),
        replacement_path,
    )
    copy_and_hash = stable_files_module._copy_and_hash
    replaced = False

    def replace_after_capture(
        source: Path,
        destination: Path,
        *,
        label: str,
        max_bytes: int | None = None,
    ) -> stable_files_module.StableFile:
        nonlocal replaced
        captured = copy_and_hash(
            source,
            destination,
            label=label,
            max_bytes=max_bytes,
        )
        if source == membership_path and not replaced:
            replaced = True
            replacement_path.replace(membership_path)
        return captured

    monkeypatch.setattr(
        stable_files_module, "_copy_and_hash", replace_after_capture
    )
    with ViewerDatabase(":memory:") as database:
        database.register_runs([descriptor])
        task_ids = database.connection.execute(
            "SELECT task_id FROM registered_tasks ORDER BY task_id"
        ).fetchall()

    assert replaced
    assert task_ids == [("CurrentTask/5",), ("CurrentTask/6",)]


def test_inexact_historical_task_schema_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "viewer.duckdb"
    _create_existing_database(path)
    with duckdb.connect(str(path)) as connection:
        _create_legacy_task_tables(connection)
        connection.execute(
            """
            INSERT INTO registered_tasks(run_id, dataset_id, task_id)
            VALUES ('run', 'dataset', 'Task/1')
            """
        )
    before = _database_snapshot(path)

    with pytest.raises(
        DatabaseSchemaError,
        match="task_annotation_tags is malformed",
    ):
        ViewerDatabase(path)

    assert _database_snapshot(path) == before


def test_task_registration_stays_bounded_for_high_cardinality_membership(
    tmp_path: Path,
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id="dataset/current",
    )
    task_ids = [f"Task/{index:05d}" for index in range(5_000)]
    membership_path = tmp_path / "high-cardinality-membership.parquet"
    descriptor = _with_membership(
        descriptor,
        membership_path,
        task_ids,
    )
    assert pq.ParquetFile(membership_path).num_row_groups > 1

    with ViewerDatabase(":memory:") as database:
        connection = database.connection

        class NoRegistrationFetchall:
            registration_select = False

            def execute(
                self,
                query: str,
                parameters: object = None,
            ) -> duckdb.DuckDBPyConnection | NoRegistrationFetchall:
                if parameters is None:
                    result = connection.execute(query)
                else:
                    result = connection.execute(query, parameters)
                self.registration_select = (
                    query.lstrip().startswith("SELECT DISTINCT")
                    and "task_id" in query
                )
                return self if self.registration_select else result

            def fetchall(self) -> object:
                if self.registration_select:
                    raise AssertionError(
                        "task registration must not fetch all task IDs"
                    )
                return connection.fetchall()

            def to_arrow_reader(self, batch_size: int) -> pa.RecordBatchReader:
                return connection.to_arrow_reader(batch_size)

        proxy = NoRegistrationFetchall()
        database._connection = cast(  # noqa: SLF001
            duckdb.DuckDBPyConnection, proxy
        )
        try:
            database.register_runs([descriptor])
        finally:
            database._connection = connection  # noqa: SLF001
        count = connection.execute(
            "SELECT count(*) FROM registered_tasks"
        ).fetchone()

    assert count == (len(task_ids),)


@pytest.mark.parametrize(
    "invalid_task_id",
    [" late-invalid ", "\tlate-invalid", "late-invalid\n"],
)
def test_task_registration_rejects_invalid_task_id_in_late_row_group(
    tmp_path: Path,
    invalid_task_id: str,
) -> None:
    descriptor = write_bundle(
        tmp_path / "bundle",
        dataset_id="dataset/current",
    )
    task_ids = [f"Task/{index:05d}" for index in range(2_000)]
    task_ids.append(invalid_task_id)
    membership_path = tmp_path / "invalid-membership.parquet"
    descriptor = _with_membership(
        descriptor,
        membership_path,
        task_ids,
    )
    assert pq.ParquetFile(membership_path).num_row_groups > 1

    with ViewerDatabase(":memory:") as database:
        with pytest.raises(
            InvalidTaskAnnotationError, match="surrounded by whitespace"
        ):
            database.register_runs([descriptor])
        count = database.connection.execute(
            "SELECT count(*) FROM registered_tasks"
        ).fetchone()

    assert count == (0,)


def test_current_schema_and_annotations_survive_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "viewer.duckdb"
    with ViewerDatabase(path) as database:
        tag = database.create_tag("Needs parser")
        database.put_annotation(
            CORPUS,
            "sample-1",
            OUTPUT,
            verdict=Verdict.SHOULD_BE_PARSEABLE,
            note="inspect",
            tag_ids=[tag.tag_id],
        )
        table_names = {
            row[0]
            for row in database.connection.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
        assert table_names == {
            *viewer_database._SCHEMA_TABLES,
            "viewer_schema",
        }
        assert database.connection.execute(
            "SELECT schema_version FROM viewer_schema"
        ).fetchall() == [(viewer_database._SCHEMA_VERSION,)]

    with ViewerDatabase(path) as reopened:
        annotation = reopened.get_annotation(CORPUS, "sample-1", OUTPUT)

    assert annotation is not None
    assert annotation.note == "inspect"
    assert annotation.verdict is Verdict.SHOULD_BE_PARSEABLE
    assert [tag.name for tag in annotation.tags] == ["Needs parser"]


def test_exact_v4_migration_preserves_every_existing_table(
    tmp_path: Path,
) -> None:
    path = tmp_path / "schema-v4.duckdb"
    task_identity = _task_identity("Task/active")
    _create_persisted_schema(
        path,
        viewer_database._SCHEMA_V4,
        viewer_database._SCHEMA_V4_VERSION,
    )
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            INSERT INTO runs(
                run_id, label, descriptor_json, manifest_sha256,
                corpus_sha256, definition_id
            ) VALUES ('run', 'Run', '{}', ?, ?, 'definition')
            """,
            ["d" * 64, CORPUS],
        )
        connection.execute(
            """
            INSERT INTO tags(tag_id, name, normalized_name)
            VALUES ('active-tag', 'Active tag', 'active tag')
            """
        )
        connection.execute(
            """
            INSERT INTO annotations(
                corpus_sha256, sample_id, decoder_output_sha256, note
            ) VALUES (?, 'active', ?, 'preserve active')
            """,
            [CORPUS, OUTPUT],
        )
        connection.execute(
            """
            INSERT INTO annotation_tags
            VALUES (?, 'active', ?, 'active-tag')
            """,
            [CORPUS, OUTPUT],
        )
        connection.execute(
            """
            INSERT INTO archived_annotations VALUES (
                ?, 'archived', ?, NULL, 'preserve archive',
                current_timestamp, current_timestamp, 'historical', 2
            )
            """,
            [CORPUS, "c" * 64],
        )
        connection.execute(
            """
            INSERT INTO archived_annotation_tags VALUES (
                ?, 'archived', ?, 'archived-tag', 'Archived tag',
                'archived tag', 'historical', 2
            )
            """,
            [CORPUS, "c" * 64],
        )
        connection.execute(
            """
            INSERT INTO archived_tags VALUES (
                'archived-tag', 'Archived tag', 'archived tag',
                current_timestamp, 'historical', 2
            )
            """
        )
        connection.execute(
            """
            INSERT INTO archived_registered_tasks
            VALUES ('archived-run', 'dataset', 'Task/archived', 'historical', 3)
            """
        )
        connection.execute(
            """
            INSERT INTO archived_task_annotations VALUES (
                'dataset', 'Task/archived', 'human', NULL, 'preserve task',
                NULL, current_timestamp, current_timestamp, 'historical', 3
            )
            """
        )
        connection.execute(
            """
            INSERT INTO archived_task_annotation_tags VALUES (
                'dataset', 'Task/archived', 'archived-task-tag',
                'Archived task tag', 'archived task tag', current_timestamp,
                'historical', 3
            )
            """
        )
        connection.execute(
            """
            INSERT INTO registered_tasks VALUES (
                'run', 'dataset', 'Task/active', ?
            )
            """,
            [task_identity],
        )
        connection.execute(
            """
            INSERT INTO task_annotations(
                dataset_id, task_id, task_identity, origin, note
            ) VALUES (
                'dataset', 'Task/active', ?, 'human', 'preserve active task'
            )
            """,
            [task_identity],
        )
        connection.execute(
            """
            INSERT INTO task_annotation_tags
            VALUES ('dataset', 'Task/active', ?, 'active-tag')
            """,
            [task_identity],
        )
        before = _table_content_signatures(
            connection, viewer_database._SCHEMA_V4_TABLES
        )

    with ViewerDatabase(path) as database:
        after = _table_content_signatures(
            database.connection, viewer_database._SCHEMA_V4_TABLES
        )
        assert database.connection.execute(
            "SELECT count(*) FROM task_annotation_publication_intents"
        ).fetchone() == (0,)
        assert database.connection.execute(
            "SELECT schema_version FROM viewer_schema"
        ).fetchall() == [(viewer_database._SCHEMA_VERSION,)]

    assert after == before


@pytest.mark.parametrize(
    "schema_version",
    [
        viewer_database._LEGACY_SCHEMA_VERSION,
        viewer_database._ANNOTATION_ARCHIVE_SCHEMA_VERSION,
    ],
)
def test_historical_publication_intent_schema_preserves_pending_intents(
    tmp_path: Path,
    schema_version: int,
) -> None:
    path = tmp_path / f"historical-intent-v{schema_version}.duckdb"
    _create_persisted_schema(
        path,
        viewer_database._AUTHENTICATED_TASK_ANNOTATION_DATABASE_WITH_PUBLICATION_INTENTS_SCHEMA,
        schema_version,
    )
    intent = _publication_intent(tmp_path)
    task_identity = _task_identity("Task/historical")
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            INSERT INTO tags(tag_id, name, normalized_name)
            VALUES ('historical-tag', 'Historical tag', 'historical tag')
            """
        )
        connection.execute(
            """
            INSERT INTO registered_tasks
            VALUES ('run', 'dataset', 'Task/historical', ?)
            """,
            [task_identity],
        )
        connection.execute(
            """
            INSERT INTO task_annotations(
                dataset_id, task_id, task_identity, origin, note
            ) VALUES (
                'dataset', 'Task/historical', ?, 'human', 'preserve task'
            )
            """,
            [task_identity],
        )
        connection.execute(
            """
            INSERT INTO task_annotation_tags
            VALUES ('dataset', 'Task/historical', ?, 'historical-tag')
            """,
            [task_identity],
        )
        connection.execute(
            """
            INSERT INTO task_annotation_publication_intents(
                producer, experiment_identity, output_path, staged_path,
                prior_sha256, intended_sha256
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                intent.producer,
                intent.experiment_identity,
                intent.output_path,
                intent.staged_path,
                intent.prior_sha256,
                intent.intended_sha256,
            ],
        )
        before = _table_content_signatures(
            connection,
            viewer_database._AUTHENTICATED_TASK_ANNOTATION_DATABASE_WITH_PUBLICATION_INTENTS_SCHEMA_TABLES,
        )

    with ViewerDatabase(path) as database:
        after = _table_content_signatures(
            database.connection,
            viewer_database._AUTHENTICATED_TASK_ANNOTATION_DATABASE_WITH_PUBLICATION_INTENTS_SCHEMA_TABLES,
        )
        assert (
            database.get_task_annotation_publication_intent(intent.output_path)
            == intent
        )
        assert database.connection.execute(
            "SELECT schema_version FROM viewer_schema"
        ).fetchone() == (viewer_database._SCHEMA_VERSION,)

    assert after == before


def test_malformed_historical_publication_intent_schema_fails_before_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "malformed-historical-intent.duckdb"
    _create_persisted_schema(
        path,
        viewer_database._AUTHENTICATED_TASK_ANNOTATION_DATABASE_WITH_PUBLICATION_INTENTS_SCHEMA,
        viewer_database._ANNOTATION_ARCHIVE_SCHEMA_VERSION,
    )
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            ALTER TABLE task_annotation_publication_intents
            ADD unexpected VARCHAR
            """
        )
    before = _database_snapshot(path)

    with pytest.raises(
        DatabaseSchemaError,
        match="historical publication-intent schema is malformed",
    ):
        ViewerDatabase(path)

    assert _database_snapshot(path) == before


def test_partial_v4_intent_addition_fails_before_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial-v4-intent.duckdb"
    _create_persisted_schema(
        path,
        viewer_database._SCHEMA_V4,
        viewer_database._SCHEMA_V4_VERSION,
    )
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            CREATE TABLE task_annotation_publication_intents (
                output_path VARCHAR PRIMARY KEY
            )
            """
        )
    before = _database_snapshot(path)

    with pytest.raises(
        DatabaseSchemaError,
        match="schema version/table signature is unsupported",
    ):
        ViewerDatabase(path)

    assert _database_snapshot(path) == before


def test_v4_migration_fault_rolls_back_intent_table_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "rollback-v4.duckdb"
    _create_persisted_schema(
        path,
        viewer_database._SCHEMA_V4,
        viewer_database._SCHEMA_V4_VERSION,
    )
    before = _database_snapshot(path)
    original_validate = viewer_database._validate_table_schema

    def fail_during_final_validation(
        connection: duckdb.DuckDBPyConnection,
        table: str,
        schema: tuple[str, ...] = viewer_database._SCHEMA,
    ) -> None:
        original_validate(connection, table, schema)
        if (
            table == "task_annotation_publication_intents"
            and schema == viewer_database._SCHEMA
        ):
            raise RuntimeError("injected v5 validation fault")

    monkeypatch.setattr(
        viewer_database,
        "_validate_table_schema",
        fail_during_final_validation,
    )

    with pytest.raises(RuntimeError, match="v5 validation fault"):
        ViewerDatabase(path)

    assert _database_snapshot(path) == before


def test_empty_database_is_initialized_and_stamped(tmp_path: Path) -> None:
    path = tmp_path / "empty.duckdb"
    with duckdb.connect(str(path)):
        pass
    before = _database_snapshot(path)

    assert before.catalog_objects == frozenset()

    with ViewerDatabase(path) as database:
        assert viewer_database._main_tables(database.connection) == (
            viewer_database._SCHEMA_TABLES | {"viewer_schema"}
        )
        assert {
            (
                catalog_object.kind,
                catalog_object.schema_name,
                catalog_object.name,
            )
            for catalog_object in viewer_database._user_catalog_objects(
                database.connection
            )
        } == {
            (viewer_database._CatalogObjectKind.TABLE, "main", table)
            for table in viewer_database._SCHEMA_TABLES | {"viewer_schema"}
        }
        assert database.connection.execute(
            "SELECT schema_version FROM viewer_schema"
        ).fetchall() == [(viewer_database._SCHEMA_VERSION,)]


def test_complete_unversioned_legacy_schema_is_admitted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.duckdb"
    _create_unversioned_schema(path)
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            INSERT INTO annotations(
                corpus_sha256, sample_id, decoder_output_sha256, verdict, note
            ) VALUES (?, 'sample-1', ?, 'expected_no_code', 'preserve me')
            """,
            [CORPUS, OUTPUT],
        )

    with ViewerDatabase(path) as database:
        annotation = database.get_annotation(CORPUS, "sample-1", OUTPUT)
        version = database.connection.execute(
            "SELECT schema_version FROM viewer_schema"
        ).fetchall()

    assert annotation is not None
    assert annotation.note == "preserve me"
    assert version == [(viewer_database._SCHEMA_VERSION,)]


def test_exact_annotation_archive_v2_migrates_without_rewriting_archive_data(
    tmp_path: Path,
) -> None:
    path = tmp_path / "annotation-archive-v2.duckdb"
    _create_persisted_schema(
        path,
        viewer_database._ANNOTATION_ARCHIVE_SCHEMA,
        viewer_database._ANNOTATION_ARCHIVE_SCHEMA_VERSION,
    )
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            INSERT INTO annotations(
                corpus_sha256, sample_id, decoder_output_sha256, note
            ) VALUES (?, 'active', ?, 'keep active')
            """,
            [CORPUS, OUTPUT],
        )
        connection.execute(
            """
            INSERT INTO archived_annotations(
                corpus_sha256, sample_id, decoder_output_sha256,
                verdict, note, created_at, updated_at, archive_reason,
                source_schema_version
            ) VALUES (
                ?, 'archived', ?, 'expected_no_code', 'keep archived',
                current_timestamp, current_timestamp, 'reason', 1
            )
            """,
            [CORPUS, "c" * 64],
        )
        connection.execute(
            """
            INSERT INTO archived_annotation_tags(
                corpus_sha256, sample_id, decoder_output_sha256,
                tag_id, tag_name, normalized_tag_name, archive_reason,
                source_schema_version
            ) VALUES (
                ?, 'archived', ?, 'archived-tag', 'Archived tag',
                'archived tag', 'reason', 1
            )
            """,
            [CORPUS, "c" * 64],
        )
        connection.execute(
            """
            INSERT INTO archived_tags(
                tag_id, name, normalized_name, created_at, archive_reason,
                source_schema_version
            ) VALUES (
                'archived-tag', 'Archived tag', 'archived tag',
                current_timestamp, 'reason', 1
            )
            """
        )

    with ViewerDatabase(path) as database:
        active = database.get_annotation(CORPUS, "active", OUTPUT)
        archive = database.connection.execute(
            """
            SELECT
                annotation.sample_id,
                link.tag_id,
                tag.name,
                annotation.archive_reason,
                annotation.source_schema_version
            FROM archived_annotations AS annotation
            JOIN archived_annotation_tags AS link USING (
                corpus_sha256, sample_id, decoder_output_sha256
            )
            JOIN archived_tags AS tag USING (tag_id)
            """
        ).fetchall()
        version = database.connection.execute(
            "SELECT schema_version FROM viewer_schema"
        ).fetchall()

    assert active is not None
    assert active.note == "keep active"
    assert archive == [
        ("archived", "archived-tag", "Archived tag", "reason", 1)
    ]
    assert version == [(viewer_database._SCHEMA_VERSION,)]


@pytest.mark.parametrize(
    "schema_version",
    [
        None,
        viewer_database._LEGACY_SCHEMA_VERSION,
        viewer_database._ANNOTATION_ARCHIVE_SCHEMA_VERSION,
    ],
)
def test_exact_pre_identity_task_schema_archives_every_task_row(
    tmp_path: Path,
    schema_version: int | None,
) -> None:
    path = tmp_path / f"task-{schema_version}.duckdb"
    created_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    updated_at = datetime(2026, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
    created_at_us = int(created_at.timestamp() * 1_000_000)
    updated_at_us = int(updated_at.timestamp() * 1_000_000)
    _create_persisted_schema(
        path,
        viewer_database._TASK_ANNOTATION_DATABASE_SCHEMA,
        schema_version,
    )
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            INSERT INTO tags(tag_id, name, normalized_name, created_at)
            VALUES ('task-tag', 'Task tag', 'task tag', ?)
            """,
            [created_at],
        )
        connection.execute(
            """
            INSERT INTO registered_tasks(run_id, dataset_id, task_id)
            VALUES ('run', 'dataset', 'Task/1')
            """
        )
        connection.execute(
            """
            INSERT INTO task_annotations(
                dataset_id, task_id, origin, category, note, provenance,
                created_at, updated_at
            ) VALUES (
                'dataset', 'Task/1', 'machine', 'review', 'keep task',
                '{"model":"legacy"}', ?, ?
            )
            """,
            [created_at, updated_at],
        )
        connection.execute(
            """
            INSERT INTO task_annotation_tags(dataset_id, task_id, tag_id)
            VALUES ('dataset', 'Task/1', 'task-tag')
            """
        )

    with ViewerDatabase(path) as database:
        active_counts = _table_counts(
            database.connection,
            (
                "registered_tasks",
                "task_annotations",
                "task_annotation_tags",
            ),
        )
        registrations = database.connection.execute(
            """
            SELECT
                run_id, dataset_id, task_id, archive_reason,
                source_schema_version
            FROM archived_registered_tasks
            """
        ).fetchall()
        annotations = database.connection.execute(
            """
            SELECT
                dataset_id, task_id, origin, category, note, provenance,
                epoch_us(created_at), epoch_us(updated_at), archive_reason,
                source_schema_version
            FROM archived_task_annotations
            """
        ).fetchall()
        links = database.connection.execute(
            """
            SELECT
                dataset_id, task_id, tag_id, tag_name, normalized_tag_name,
                epoch_us(tag_created_at), archive_reason,
                source_schema_version
            FROM archived_task_annotation_tags
            """
        ).fetchall()
        version = database.connection.execute(
            "SELECT schema_version FROM viewer_schema"
        ).fetchall()

    source_version = schema_version or viewer_database._LEGACY_SCHEMA_VERSION
    reason = "missing_authenticated_task_identity"
    assert active_counts == (0, 0, 0)
    assert registrations == [
        ("run", "dataset", "Task/1", reason, source_version)
    ]
    assert annotations == [
        (
            "dataset",
            "Task/1",
            "machine",
            "review",
            "keep task",
            '{"model":"legacy"}',
            created_at_us,
            updated_at_us,
            reason,
            source_version,
        )
    ]
    assert links == [
        (
            "dataset",
            "Task/1",
            "task-tag",
            "Task tag",
            "task tag",
            created_at_us,
            reason,
            source_version,
        )
    ]
    assert version == [(viewer_database._SCHEMA_VERSION,)]


def test_exact_v3_archives_pre_identity_tasks_and_preserves_archive_data(
    tmp_path: Path,
) -> None:
    path = tmp_path / "schema-v3.duckdb"
    timestamp = datetime(2026, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
    timestamp_us = int(timestamp.timestamp() * 1_000_000)
    _create_persisted_schema(
        path,
        viewer_database._SCHEMA_V3,
        viewer_database._PRE_IDENTITY_TASK_SCHEMA_VERSION,
    )
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            INSERT INTO archived_annotations(
                corpus_sha256, sample_id, decoder_output_sha256,
                verdict, note, created_at, updated_at, archive_reason,
                source_schema_version
            ) VALUES (
                ?, 'already-archived', ?, NULL, 'preserve archive', ?, ?,
                'historical_reason', 2
            )
            """,
            [CORPUS, OUTPUT, timestamp, timestamp],
        )
        connection.execute(
            """
            INSERT INTO registered_tasks(run_id, dataset_id, task_id)
            VALUES ('run-v3', 'dataset', 'Task/v3')
            """
        )
        connection.execute(
            """
            INSERT INTO task_annotations(
                dataset_id, task_id, origin, category, note, provenance,
                created_at, updated_at
            ) VALUES (
                'dataset', 'Task/v3', 'human', NULL, 'preserve task', NULL,
                ?, ?
            )
            """,
            [timestamp, timestamp],
        )

    with ViewerDatabase(path) as database:
        archived_annotation = database.connection.execute(
            """
            SELECT
                corpus_sha256, sample_id, decoder_output_sha256,
                verdict, note, epoch_us(created_at), epoch_us(updated_at),
                archive_reason, source_schema_version
            FROM archived_annotations
            """
        ).fetchall()
        archived_task = database.connection.execute(
            """
            SELECT
                dataset_id, task_id, origin, category, note, provenance,
                epoch_us(created_at), epoch_us(updated_at), archive_reason,
                source_schema_version
            FROM archived_task_annotations
            """
        ).fetchall()
        archived_registration = database.connection.execute(
            """
            SELECT run_id, dataset_id, task_id, source_schema_version
            FROM archived_registered_tasks
            """
        ).fetchall()
        version = database.connection.execute(
            "SELECT schema_version FROM viewer_schema"
        ).fetchall()

    assert archived_annotation == [
        (
            CORPUS,
            "already-archived",
            OUTPUT,
            None,
            "preserve archive",
            timestamp_us,
            timestamp_us,
            "historical_reason",
            2,
        )
    ]
    assert archived_task == [
        (
            "dataset",
            "Task/v3",
            "human",
            None,
            "preserve task",
            None,
            timestamp_us,
            timestamp_us,
            "missing_authenticated_task_identity",
            viewer_database._PRE_IDENTITY_TASK_SCHEMA_VERSION,
        )
    ]
    assert archived_registration == [
        (
            "run-v3",
            "dataset",
            "Task/v3",
            viewer_database._PRE_IDENTITY_TASK_SCHEMA_VERSION,
        )
    ]
    assert version == [(viewer_database._SCHEMA_VERSION,)]


@pytest.mark.parametrize(
    "schema_version",
    [
        viewer_database._LEGACY_SCHEMA_VERSION,
        viewer_database._ANNOTATION_ARCHIVE_SCHEMA_VERSION,
    ],
)
def test_historical_authenticated_task_signature_preserves_active_rows(
    tmp_path: Path,
    schema_version: int,
) -> None:
    path = tmp_path / f"authenticated-task-{schema_version}.duckdb"
    task_identity = _task_identity("Task/1")
    _create_persisted_schema(
        path,
        viewer_database._AUTHENTICATED_TASK_ANNOTATION_DATABASE_SCHEMA,
        schema_version,
    )
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            INSERT INTO tags(tag_id, name, normalized_name)
            VALUES ('task-tag', 'Task tag', 'task tag')
            """
        )
        connection.execute(
            """
            INSERT INTO registered_tasks(
                run_id, dataset_id, task_id, task_identity
            ) VALUES ('run', 'dataset', 'Task/1', ?)
            """,
            [task_identity],
        )
        connection.execute(
            """
            INSERT INTO task_annotations(
                dataset_id, task_id, task_identity, origin, category, note
            ) VALUES ('dataset', 'Task/1', ?, 'human', 'review', 'keep task')
            """,
            [task_identity],
        )
        connection.execute(
            """
            INSERT INTO task_annotation_tags(
                dataset_id, task_id, task_identity, tag_id
            ) VALUES ('dataset', 'Task/1', ?, 'task-tag')
            """,
            [task_identity],
        )

    with ViewerDatabase(path) as database:
        annotation = database.get_task_annotation(
            "dataset", "Task/1", task_identity
        )
        registrations = database.connection.execute(
            """
            SELECT run_id, dataset_id, task_id, task_identity
            FROM registered_tasks
            """
        ).fetchall()
        archive_count = database.connection.execute(
            "SELECT count(*) FROM archived_task_annotations"
        ).fetchone()
        version = database.connection.execute(
            "SELECT schema_version FROM viewer_schema"
        ).fetchall()

    assert annotation is not None
    assert annotation.note == "keep task"
    assert [tag.name for tag in annotation.tags] == ["Task tag"]
    assert registrations == [("run", "dataset", "Task/1", task_identity)]
    assert archive_count == (0,)
    assert version == [(viewer_database._SCHEMA_VERSION,)]


def test_malformed_v3_task_signature_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "malformed-v3.duckdb"
    _create_persisted_schema(
        path,
        viewer_database._SCHEMA_V3,
        viewer_database._PRE_IDENTITY_TASK_SCHEMA_VERSION,
    )
    with duckdb.connect(str(path)) as connection:
        connection.execute("ALTER TABLE task_annotations ADD extra VARCHAR")
    before = _database_snapshot(path)

    with pytest.raises(
        DatabaseSchemaError,
        match="historical composed schema is malformed",
    ):
        ViewerDatabase(path)

    assert _database_snapshot(path) == before


def test_malformed_annotation_archive_v2_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "malformed-v2.duckdb"
    _create_persisted_schema(
        path,
        viewer_database._ANNOTATION_ARCHIVE_SCHEMA,
        viewer_database._ANNOTATION_ARCHIVE_SCHEMA_VERSION,
    )
    with duckdb.connect(str(path)) as connection:
        connection.execute("ALTER TABLE archived_tags ADD extra VARCHAR")
    before = _database_snapshot(path)

    with pytest.raises(
        DatabaseSchemaError, match="archived_tags is malformed"
    ):
        ViewerDatabase(path)

    assert _database_snapshot(path) == before


def test_annotation_archive_v2_fault_rolls_back_schema_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "rollback-v2.duckdb"
    _create_persisted_schema(
        path,
        viewer_database._ANNOTATION_ARCHIVE_SCHEMA,
        viewer_database._ANNOTATION_ARCHIVE_SCHEMA_VERSION,
    )
    before = _database_snapshot(path)
    original_validate = viewer_database._validate_table_schema

    def fail_during_final_validation(
        connection: duckdb.DuckDBPyConnection,
        table: str,
        schema: tuple[str, ...] = viewer_database._SCHEMA,
    ) -> None:
        original_validate(connection, table, schema)
        if table == "task_annotations" and schema == viewer_database._SCHEMA:
            raise RuntimeError("injected final-schema validation fault")

    monkeypatch.setattr(
        viewer_database,
        "_validate_table_schema",
        fail_during_final_validation,
    )

    with pytest.raises(RuntimeError, match="final-schema validation fault"):
        ViewerDatabase(path)

    assert _database_snapshot(path) == before


def test_v3_task_archive_fault_rolls_back_rows_schema_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "rollback-v3.duckdb"
    _create_persisted_schema(
        path,
        viewer_database._SCHEMA_V3,
        viewer_database._PRE_IDENTITY_TASK_SCHEMA_VERSION,
    )
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            INSERT INTO registered_tasks(run_id, dataset_id, task_id)
            VALUES ('run', 'dataset', 'Task/1')
            """
        )
        connection.execute(
            """
            INSERT INTO task_annotations(dataset_id, task_id, origin, note)
            VALUES ('dataset', 'Task/1', 'human', 'must roll back')
            """
        )
    before = _database_snapshot(path)
    original_validate = viewer_database._validate_table_schema

    def fail_during_final_validation(
        connection: duckdb.DuckDBPyConnection,
        table: str,
        schema: tuple[str, ...] = viewer_database._SCHEMA,
    ) -> None:
        original_validate(connection, table, schema)
        if (
            table == "archived_task_annotations"
            and schema == viewer_database._SCHEMA
        ):
            raise RuntimeError("injected task-archive validation fault")

    monkeypatch.setattr(
        viewer_database,
        "_validate_table_schema",
        fail_during_final_validation,
    )

    with pytest.raises(RuntimeError, match="task-archive validation fault"):
        ViewerDatabase(path)

    assert _database_snapshot(path) == before


def test_pre_identity_task_tag_snapshot_survives_tag_archival(
    tmp_path: Path,
) -> None:
    path = tmp_path / "task-tag-conflict.duckdb"
    _create_persisted_schema(
        path,
        viewer_database._TASK_ANNOTATION_DATABASE_SCHEMA,
        viewer_database._ANNOTATION_ARCHIVE_SCHEMA_VERSION,
    )
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            INSERT INTO tags(tag_id, name, normalized_name)
            VALUES ('invalid-tag', ?, ?)
            """,
            ["x" * 101, "x" * 101],
        )
        connection.execute(
            """
            INSERT INTO task_annotations(dataset_id, task_id, origin)
            VALUES ('dataset', 'Task/1', 'human')
            """
        )
        connection.execute(
            """
            INSERT INTO task_annotation_tags(dataset_id, task_id, tag_id)
            VALUES ('dataset', 'Task/1', 'invalid-tag')
            """
        )

    with ViewerDatabase(path) as database:
        archived_link = database.connection.execute(
            """
            SELECT
                dataset_id, task_id, tag_id, tag_name, normalized_tag_name,
                archive_reason, source_schema_version
            FROM archived_task_annotation_tags
            """
        ).fetchall()
        archived_task = database.connection.execute(
            """
            SELECT dataset_id, task_id, archive_reason, source_schema_version
            FROM archived_task_annotations
            """
        ).fetchall()
        archived_tag = database.connection.execute(
            "SELECT tag_id, name, archive_reason FROM archived_tags"
        ).fetchall()
        active_counts = _table_counts(
            database.connection,
            (
                "tags",
                "task_annotations",
                "task_annotation_tags",
            ),
        )

    reason = "missing_authenticated_task_identity"
    assert archived_link == [
        (
            "dataset",
            "Task/1",
            "invalid-tag",
            "x" * 101,
            "x" * 101,
            reason,
            viewer_database._ANNOTATION_ARCHIVE_SCHEMA_VERSION,
        )
    ]
    assert archived_task == [
        (
            "dataset",
            "Task/1",
            reason,
            viewer_database._ANNOTATION_ARCHIVE_SCHEMA_VERSION,
        )
    ]
    assert archived_tag == [
        ("invalid-tag", "x" * 101, "tag_name_is_out_of_contract")
    ]
    assert active_counts == (0, 0, 0)


def test_v1_outliers_are_archived_with_complete_annotation_closure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-v1.duckdb"
    _create_unversioned_schema(path)
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            "CREATE TABLE viewer_schema (schema_version INTEGER PRIMARY KEY)"
        )
        connection.execute("INSERT INTO viewer_schema VALUES (1)")
        tag_rows = [
            (f"tag-{index:03}", f"tag {index:03}", f"tag {index:03}")
            for index in range(101)
        ]
        tag_rows.append(("invalid-tag", "x" * 101, "x" * 101))
        connection.executemany(
            "INSERT INTO tags(tag_id, name, normalized_name) VALUES (?, ?, ?)",
            tag_rows,
        )
        connection.executemany(
            """
            INSERT INTO annotations(
                corpus_sha256, sample_id, decoder_output_sha256, note
            ) VALUES (?, ?, ?, ?)
            """,
            [
                (CORPUS, "long-note", OUTPUT, "n" * 10_001),
                (CORPUS, "many-tags", "c" * 64, "many"),
                (CORPUS, "invalid-tag", "d" * 64, "tag closure"),
            ],
        )
        connection.executemany(
            "INSERT INTO annotation_tags VALUES (?, ?, ?, ?)",
            [
                (CORPUS, "many-tags", "c" * 64, tag_id)
                for tag_id, _name, _normalized_name in tag_rows[:101]
            ]
            + [
                (CORPUS, "invalid-tag", "d" * 64, "tag-000"),
                (CORPUS, "invalid-tag", "d" * 64, "invalid-tag"),
            ],
        )

    with ViewerDatabase(path) as database:
        assert database.connection.execute(
            "SELECT schema_version FROM viewer_schema"
        ).fetchall() == [(viewer_database._SCHEMA_VERSION,)]
        assert database.connection.execute(
            "SELECT sample_id, archive_reason FROM archived_annotations "
            "ORDER BY sample_id"
        ).fetchall() == [
            ("invalid-tag", "references_archived_tag"),
            ("long-note", "note_exceeds_max_length"),
            ("many-tags", "tag_count_exceeds_maximum"),
        ]
        assert database.connection.execute(
            """
            SELECT tag_id, tag_name
            FROM archived_annotation_tags
            WHERE sample_id = 'invalid-tag'
            ORDER BY tag_id
            """
        ).fetchall() == [
            ("invalid-tag", "x" * 101),
            ("tag-000", "tag 000"),
        ]
        assert database.connection.execute(
            "SELECT tag_id, archive_reason FROM archived_tags"
        ).fetchall() == [("invalid-tag", "tag_name_is_out_of_contract")]
        assert database.connection.execute(
            "SELECT count(*) FROM archived_annotation_tags"
        ).fetchone() == (103,)
        assert database.connection.execute(
            "SELECT count(*) FROM annotations"
        ).fetchone() == (0,)
        assert database.connection.execute(
            "SELECT count(*) FROM annotation_tags"
        ).fetchone() == (0,)


def test_current_outlier_is_archived_before_active_reads(
    tmp_path: Path,
) -> None:
    path = tmp_path / "current.duckdb"
    with ViewerDatabase(path):
        pass
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            INSERT INTO annotations(
                corpus_sha256, sample_id, decoder_output_sha256, note
            ) VALUES (?, 'outlier', ?, ?)
            """,
            [CORPUS, OUTPUT, "n" * 10_001],
        )

    with ViewerDatabase(path) as database:
        assert database.get_annotation(CORPUS, "outlier", OUTPUT) is None
        assert database.connection.execute(
            """
            SELECT sample_id, archive_reason, source_schema_version
            FROM archived_annotations
            """
        ).fetchall() == [
            (
                "outlier",
                "note_exceeds_max_length",
                viewer_database._SCHEMA_VERSION,
            )
        ]


def test_annotation_contract_accepts_maxima_and_rejects_max_plus_one() -> None:
    with ViewerDatabase(":memory:") as database:
        tags = [database.create_tag(f"tag {index:03}") for index in range(101)]
        accepted = database.put_annotation(
            CORPUS,
            "sample-1",
            OUTPUT,
            verdict=None,
            note="n" * 10_000,
            tag_ids=[tag.tag_id for tag in tags[:100]] + [tags[0].tag_id],
        )

        assert len(accepted.note or "") == 10_000
        assert len(accepted.tags) == 100
        with pytest.raises(InvalidQueryError, match="100 distinct tag IDs"):
            database.put_annotation(
                CORPUS,
                "sample-1",
                OUTPUT,
                verdict=None,
                note="replacement",
                tag_ids=[tag.tag_id for tag in tags],
            )
        with pytest.raises(InvalidQueryError, match="10000 characters"):
            database.put_annotation(
                CORPUS,
                "sample-1",
                OUTPUT,
                verdict=None,
                note="n" * 10_001,
            )

        unchanged = database.get_annotation(CORPUS, "sample-1", OUTPUT)
        assert unchanged == accepted


def test_tag_contract_applies_after_whitespace_normalization() -> None:
    with ViewerDatabase(":memory:") as database:
        tag = database.create_tag("  " + "x" * 100 + "  ")
        normalized_short = database.create_tag("left" + " " * 200 + "right")

        assert tag.name == "x" * 100
        assert normalized_short.name == "left right"
        with pytest.raises(InvalidQueryError, match="after normalization"):
            database.create_tag("x" * 101)


def test_partial_unversioned_schema_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute(viewer_database._SCHEMA[0])
    before = _database_snapshot(path)

    with pytest.raises(DatabaseSchemaError, match="unversioned schema"):
        ViewerDatabase(path)

    assert _database_snapshot(path) == before
    assert before.version is None


def test_unknown_unversioned_schema_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unknown.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE mystery (value INTEGER)")
    before = _database_snapshot(path)

    with pytest.raises(DatabaseSchemaError, match="unexpected mystery"):
        ViewerDatabase(path)

    assert _database_snapshot(path) == before
    assert before.version is None


def test_malformed_unversioned_schema_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "malformed.duckdb"
    _create_unversioned_schema(path)
    with duckdb.connect(str(path)) as connection:
        connection.execute("DROP TABLE annotations")
        connection.execute(
            "CREATE TABLE annotations (sample_id VARCHAR PRIMARY KEY)"
        )
    before = _database_snapshot(path)

    with pytest.raises(DatabaseSchemaError, match="annotations is malformed"):
        ViewerDatabase(path)

    assert _database_snapshot(path) == before
    assert before.version is None


def test_unknown_sequence_is_rejected_without_mutation(tmp_path: Path) -> None:
    path = tmp_path / "unknown-sequence.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE SEQUENCE unrelated_sequence START 42")
    before = _database_snapshot(path)

    with pytest.raises(
        DatabaseSchemaError,
        match=r"sequence .*\.main\.unrelated_sequence",
    ):
        ViewerDatabase(path)

    assert _database_snapshot(path) == before
    with duckdb.connect(str(path), read_only=True) as connection:
        assert connection.execute(
            """
            SELECT sequence_name, start_value, last_value
            FROM duckdb_sequences()
            """
        ).fetchall() == [("unrelated_sequence", 42, None)]


@pytest.mark.parametrize(
    ("object_kind", "object_name", "statements"),
    [
        (
            "view",
            "unrelated_view",
            ("CREATE VIEW unrelated_view AS SELECT 42 AS value",),
        ),
        (
            "schema",
            "unrelated_schema",
            ("CREATE SCHEMA unrelated_schema",),
        ),
        (
            "type",
            "unrelated_type",
            ("CREATE TYPE unrelated_type AS ENUM ('left', 'right')",),
        ),
        (
            "function",
            "unrelated_macro",
            ("CREATE MACRO unrelated_macro(value) AS value + 1",),
        ),
        (
            "index",
            "unrelated_index",
            (
                "CREATE TABLE indexed_values (value INTEGER)",
                "CREATE INDEX unrelated_index ON indexed_values(value)",
            ),
        ),
    ],
)
def test_unknown_persistent_catalog_object_is_rejected_without_mutation(
    tmp_path: Path,
    object_kind: str,
    object_name: str,
    statements: tuple[str, ...],
) -> None:
    path = tmp_path / f"unknown-{object_kind}.duckdb"
    with duckdb.connect(str(path)) as connection:
        for statement in statements:
            connection.execute(statement)
    before = _database_snapshot(path)

    with pytest.raises(
        DatabaseSchemaError,
        match=rf"{object_kind} .*\.{object_name}",
    ):
        ViewerDatabase(path)

    assert _database_snapshot(path) == before
    assert any(
        catalog_object.kind == object_kind
        and catalog_object.name == object_name
        for catalog_object in before.catalog_objects
    )


def test_archive_reason_literals_and_unicode_scalar_classification() -> None:
    assert {
        reason.name: reason.value for reason in viewer_database._ArchiveReason
    } == {
        "MISSING_AUTHENTICATED_TASK_IDENTITY": (
            "missing_authenticated_task_identity"
        ),
        "NOTE_EXCEEDS_MAX_LENGTH": "note_exceeds_max_length",
        "NOTE_IS_NOT_UNICODE_SCALAR": "note_is_not_unicode_scalar",
        "NORMALIZED_TAG_NAME_IS_NOT_CANONICAL": (
            "normalized_tag_name_is_not_canonical"
        ),
        "REFERENCES_ARCHIVED_TAG": "references_archived_tag",
        "TAG_COUNT_EXCEEDS_MAXIMUM": "tag_count_exceeds_maximum",
        "TAG_NAME_IS_MALFORMED": "tag_name_is_malformed",
        "TAG_NAME_IS_NOT_NORMALIZED": "tag_name_is_not_normalized",
        "TAG_NAME_IS_OUT_OF_CONTRACT": "tag_name_is_out_of_contract",
    }
    assert viewer_database._annotation_note_archive_reason("\ud800") is (
        viewer_database._ArchiveReason.NOTE_IS_NOT_UNICODE_SCALAR
    )
    assert viewer_database._annotation_note_archive_reason("n" * 10_001) is (
        viewer_database._ArchiveReason.NOTE_EXCEEDS_MAX_LENGTH
    )
    assert viewer_database._tag_archive_reasons("\ud800", "\ud800") == (
        viewer_database._ArchiveReason.TAG_NAME_IS_OUT_OF_CONTRACT,
    )


def test_annotation_identity_includes_exact_output_hash() -> None:
    with ViewerDatabase(":memory:") as database:
        database.put_annotation(
            CORPUS,
            "sample-1",
            OUTPUT,
            verdict=None,
            note="first",
        )

        assert database.get_annotation(CORPUS, "sample-1", "c" * 64) is None


def test_annotation_export_is_deterministic() -> None:
    with ViewerDatabase(":memory:") as database:
        second = database.create_tag("zeta")
        first = database.create_tag("Alpha")
        database.put_annotation(
            CORPUS,
            "sample-2",
            "d" * 64,
            verdict=None,
            note="note",
            tag_ids=[second.tag_id, first.tag_id],
        )
        database.put_annotation(
            CORPUS,
            "sample-1",
            OUTPUT,
            verdict=Verdict.EXPECTED_NO_CODE,
            note=None,
        )

        exported = database.export_annotations()

    assert [item["sample_id"] for item in exported] == [
        "sample-1",
        "sample-2",
    ]
    assert exported[1]["tags"] == ["Alpha", "zeta"]
    assert not any("created_at" in item for item in exported)


def test_unknown_tag_rejects_annotation_atomically() -> None:
    with ViewerDatabase(":memory:") as database:
        with pytest.raises(InvalidQueryError, match="unknown tag"):
            database.put_annotation(
                CORPUS,
                "sample-1",
                OUTPUT,
                verdict=None,
                note="must roll back",
                tag_ids=["missing"],
            )

        assert database.get_annotation(CORPUS, "sample-1", OUTPUT) is None


def test_delete_annotation_removes_tag_links() -> None:
    with ViewerDatabase(":memory:") as database:
        tag = database.create_tag("parser")
        database.put_annotation(
            CORPUS,
            "sample-1",
            OUTPUT,
            verdict=None,
            note=None,
            tag_ids=[tag.tag_id],
        )

        assert database.delete_annotation(CORPUS, "sample-1", OUTPUT)
        assert database.get_annotation(CORPUS, "sample-1", OUTPUT) is None
        assert database.connection.execute(
            "SELECT count(*) FROM annotation_tags"
        ).fetchone() == (0,)


def test_high_contention_constructors_share_initialized_database(
    tmp_path: Path,
) -> None:
    path = tmp_path / "viewer.duckdb"
    workers = 32
    start = threading.Barrier(workers)
    release = threading.Event()
    state_changed = threading.Condition()
    opened = 0
    errors: list[BaseException] = []

    def construct(_index: int) -> set[str]:
        nonlocal opened
        start.wait()
        constructor_path = (
            path if _index % 2 else tmp_path / "path-alias" / ".." / path.name
        )
        try:
            database = ViewerDatabase(constructor_path)
        except BaseException as exc:
            with state_changed:
                errors.append(exc)
                state_changed.notify()
            raise
        try:
            with state_changed:
                opened += 1
                state_changed.notify()
            assert release.wait(timeout=30)
            return {
                row[0]
                for row in database.connection.execute(
                    "SELECT table_name FROM information_schema.tables"
                ).fetchall()
            }
        finally:
            database.close()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(construct, index) for index in range(workers)
        ]
        with state_changed:
            constructors_finished = state_changed.wait_for(
                lambda: opened + len(errors) == workers,
                timeout=30,
            )
        release.set()
        schemas = [future.result() for future in futures]

    assert constructors_finished
    assert errors == []
    expected_tables = {*viewer_database._SCHEMA_TABLES, "viewer_schema"}
    assert schemas == [expected_tables] * workers


def test_constructor_failure_closes_connection_and_releases_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "viewer.duckdb"
    real_connect = viewer_database.duckdb.connect
    real_create_schema = viewer_database._create_schema
    opened_connections = []

    def recording_connect(raw_path: str):
        connection = real_connect(raw_path)
        opened_connections.append(connection)
        return connection

    def fail_schema(_connection) -> None:
        raise RuntimeError("schema failure")

    monkeypatch.setattr(viewer_database.duckdb, "connect", recording_connect)
    monkeypatch.setattr(viewer_database, "_create_schema", fail_schema)

    with pytest.raises(RuntimeError, match="schema failure"):
        ViewerDatabase(path)

    assert len(opened_connections) == 1
    with pytest.raises(
        viewer_database.duckdb.ConnectionException,
        match="already closed",
    ):
        opened_connections[0].execute("SELECT 1")

    monkeypatch.setattr(viewer_database, "_create_schema", real_create_schema)
    with ViewerDatabase(path) as database:
        assert database.connection.execute("SELECT 1").fetchone() == (1,)


def test_connect_binder_failure_releases_guards_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "viewer.duckdb"
    real_connect = viewer_database.duckdb.connect
    attempts = 0

    def fail_once(raw_path: str):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise viewer_database.duckdb.BinderException("connect conflict")
        return real_connect(raw_path)

    monkeypatch.setattr(viewer_database.duckdb, "connect", fail_once)

    with pytest.raises(
        viewer_database.duckdb.BinderException,
        match="connect conflict",
    ):
        ViewerDatabase(path)
    assert attempts == 1

    with ViewerDatabase(path) as database:
        assert database.connection.execute("SELECT 1").fetchone() == (1,)
    assert attempts == 2


def test_other_process_cannot_open_owned_mutable_database(
    tmp_path: Path,
) -> None:
    path = tmp_path / "viewer.duckdb"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "from dr_code.viewer.database import ViewerDatabase; "
                f"database = ViewerDatabase({str(path)!r}); "
                "print('ready', flush=True); "
                "input(); "
                "database.close()"
            ),
        ],
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert child.stdout is not None
    assert child.stdin is not None
    try:
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(
            DatabaseOwnershipError,
            match="already owned by another process",
        ):
            ViewerDatabase(path)
    finally:
        child.stdin.write("\n")
        child.stdin.flush()
        _, stderr = child.communicate(timeout=10)
    assert child.returncode == 0, stderr


@pytest.mark.skipif(
    not hasattr(__import__("os"), "fork"), reason="requires fork"
)
def test_inherited_database_is_fork_safe_and_parent_retains_ownership(
    tmp_path: Path,
) -> None:
    path = tmp_path / "viewer.duckdb"
    script = r"""
import os
from pathlib import Path
import subprocess
import sys
import threading

import dr_code.viewer.database as module
from dr_code.viewer.database import DatabaseOwnershipError, ViewerDatabase

path = Path(sys.argv[1])
database = ViewerDatabase(path)
guards_held = threading.Barrier(3)
release = threading.Event()

def hold(lock):
    lock.acquire()
    try:
        guards_held.wait()
        release.wait()
    finally:
        lock.release()

threads = [
    threading.Thread(target=hold, args=(module._OWNERSHIP_GUARD,), daemon=True),
    threading.Thread(target=hold, args=(module._INITIALIZATION_GUARD,), daemon=True),
]
for thread in threads:
    thread.start()
guards_held.wait()

try:
    pid = os.fork()
    if pid == 0:
        try:
            try:
                database.connection.execute("SELECT 1")
            except DatabaseOwnershipError:
                pass
            else:
                os._exit(10)
            database.close()
            try:
                ViewerDatabase(path)
            except DatabaseOwnershipError:
                os._exit(0)
            os._exit(11)
        except BaseException:
            os._exit(12)
finally:
    release.set()
    for thread in threads:
        thread.join()

_, status = os.waitpid(pid, 0)
if os.waitstatus_to_exitcode(status) != 0:
    raise SystemExit(20 + os.waitstatus_to_exitcode(status))

contender = subprocess.run(
    [
        sys.executable,
        "-c",
        "from dr_code.viewer.database import ViewerDatabase; "
        "import sys; ViewerDatabase(sys.argv[1])",
        str(path),
    ],
    capture_output=True,
    text=True,
)
if contender.returncode == 0 or "already owned" not in contender.stderr:
    raise SystemExit(30)
if database.connection.execute("SELECT 42").fetchone() != (42,):
    raise SystemExit(31)
database.close()
print("fork-safe", flush=True)
"""
    result = run_subprocess(
        command=(sys.executable, "-c", script, str(path)),
        input_text="",
        timeout_seconds=20.0,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fork-safe"


@pytest.mark.skipif(
    not hasattr(__import__("os"), "fork"), reason="requires fork"
)
def test_inherited_database_rejects_serialization_before_inherited_mutex(
    tmp_path: Path,
) -> None:
    path = tmp_path / "viewer.duckdb"
    script = r"""
import os
from pathlib import Path
import sys
import threading

from dr_code.viewer.database import DatabaseOwnershipError, ViewerDatabase

database = ViewerDatabase(Path(sys.argv[1]))
locked = threading.Event()
release = threading.Event()

def hold_mutex():
    with database._classification_mutex:
        locked.set()
        release.wait()

thread = threading.Thread(target=hold_mutex, daemon=True)
thread.start()
if not locked.wait(timeout=2):
    raise SystemExit(10)

pid = os.fork()
if pid == 0:
    try:
        with database.classification_serialization():
            os._exit(11)
    except DatabaseOwnershipError:
        os._exit(0)
    except BaseException:
        os._exit(12)

try:
    _, status = os.waitpid(pid, 0)
    exit_code = os.waitstatus_to_exitcode(status)
    if exit_code != 0:
        raise SystemExit(20 + exit_code)
finally:
    release.set()
    thread.join(timeout=2)
    database.close()
print("inherited serialization rejected", flush=True)
"""
    result = run_subprocess(
        command=(sys.executable, "-c", script, str(path)),
        input_text="",
        timeout_seconds=10.0,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "inherited serialization rejected"
