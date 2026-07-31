from __future__ import annotations

from dataclasses import replace
import hashlib
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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


def test_database_owner_lock_path_is_canonical(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / ".." / "viewer.duckdb"

    assert (
        database_owner_lock_path(database_path)
        == (tmp_path / ".viewer.duckdb.owner.lock").resolve()
    )
    with pytest.raises(ValueError, match="no ownership lock"):
        database_owner_lock_path(":memory:")


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
    return replace(descriptor, candidate_membership_path=path)


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
            provenance=_machine_candidate(
                "dataset", "Task/1", category="matching"
            ).provenance,
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
            provenance=_machine_candidate(
                "dataset", "Task/1", category="machine"
            ).provenance,
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
            provenance=_machine_candidate(
                "dataset", "Task/1", category="single"
            ).provenance,
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


def test_legacy_registration_migrates_without_ambiguous_annotations(
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

    with ViewerDatabase(path) as database:
        columns = {
            row[1]
            for row in database.connection.execute(
                "PRAGMA table_info('task_annotations')"
            ).fetchall()
        }
        version = database.connection.execute(
            "SELECT schema_version FROM viewer_schema"
        ).fetchone()
        registrations = database.connection.execute(
            "SELECT count(*) FROM registered_tasks"
        ).fetchone()

    assert "task_identity" in columns
    assert registrations == (0,)
    assert version == (1,)


def test_legacy_task_rows_without_content_identity_fail_fast(
    tmp_path: Path,
) -> None:
    path = tmp_path / "viewer.duckdb"
    _create_existing_database(path)
    with duckdb.connect(str(path)) as connection:
        _create_legacy_task_tables(connection)
        connection.execute(
            """
            INSERT INTO task_annotations(dataset_id, task_id, origin)
            VALUES ('dataset', 'Task/1', 'human')
            """
        )

    with pytest.raises(
        DatabaseSchemaError,
        match="lack authenticated task_identity",
    ):
        ViewerDatabase(path)


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
            "annotation_tags",
            "annotations",
            "registered_tasks",
            "runs",
            "tags",
            "task_annotation_tags",
            "task_annotation_publication_intents",
            "task_annotations",
            "viewer_schema",
        }

    with ViewerDatabase(path) as reopened:
        annotation = reopened.get_annotation(CORPUS, "sample-1", OUTPUT)

    assert annotation is not None
    assert annotation.note == "inspect"
    assert annotation.verdict is Verdict.SHOULD_BE_PARSEABLE
    assert [tag.name for tag in annotation.tags] == ["Needs parser"]


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
    expected_tables = {
        "annotation_tags",
        "annotations",
        "registered_tasks",
        "runs",
        "tags",
        "task_annotation_tags",
        "task_annotation_publication_intents",
        "task_annotations",
        "viewer_schema",
    }
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
