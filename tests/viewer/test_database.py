from __future__ import annotations

import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import dr_code.viewer.database as viewer_database
from dr_code.viewer.database import (
    DatabaseOwnershipError,
    ViewerDatabase,
    database_owner_lock_path,
)
from dr_code.viewer.domain import InvalidQueryError, Verdict

CORPUS = "a" * 64
OUTPUT = "b" * 64


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
            "runs",
            "tags",
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
    expected_tables = {"annotation_tags", "annotations", "runs", "tags"}
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
