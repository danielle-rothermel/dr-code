from __future__ import annotations

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pytest

from dr_code.execution import run_subprocess
import dr_code.viewer.database as viewer_database
from dr_code.viewer.database import (
    DatabaseOwnershipError,
    DatabaseSchemaError,
    ViewerDatabase,
    database_owner_lock_path,
)
from dr_code.viewer.domain import InvalidQueryError, Verdict

CORPUS = "a" * 64
OUTPUT = "b" * 64


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


def test_database_owner_lock_path_is_canonical(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / ".." / "viewer.duckdb"

    assert (
        database_owner_lock_path(database_path)
        == (tmp_path / ".viewer.duckdb.owner.lock").resolve()
    )
    with pytest.raises(ValueError, match="no ownership lock"):
        database_owner_lock_path(":memory:")


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
            "archived_annotation_tags",
            "archived_annotations",
            "archived_tags",
            "runs",
            "tags",
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


def test_v2_outlier_is_archived_before_active_reads(tmp_path: Path) -> None:
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
