"""DuckDB run catalog and example-annotation persistence."""

from __future__ import annotations

import errno
import fcntl
import os
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

import duckdb

from dr_code.viewer.domain import (
    Annotation,
    InvalidQueryError,
    RunDescriptor,
    Tag,
    Verdict,
    validate_sha256,
)


_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id VARCHAR PRIMARY KEY,
        label VARCHAR NOT NULL,
        descriptor_json VARCHAR NOT NULL,
        manifest_sha256 VARCHAR NOT NULL,
        corpus_sha256 VARCHAR NOT NULL,
        definition_id VARCHAR NOT NULL,
        registered_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS annotations (
        corpus_sha256 VARCHAR NOT NULL,
        sample_id VARCHAR NOT NULL,
        decoder_output_sha256 VARCHAR NOT NULL,
        verdict VARCHAR CHECK (
            verdict IN ('should_be_parseable', 'expected_no_code')
        ),
        note VARCHAR,
        created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
        PRIMARY KEY (corpus_sha256, sample_id, decoder_output_sha256)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tags (
        tag_id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        normalized_name VARCHAR NOT NULL UNIQUE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS annotation_tags (
        corpus_sha256 VARCHAR NOT NULL,
        sample_id VARCHAR NOT NULL,
        decoder_output_sha256 VARCHAR NOT NULL,
        tag_id VARCHAR NOT NULL REFERENCES tags(tag_id),
        PRIMARY KEY (
            corpus_sha256, sample_id, decoder_output_sha256, tag_id
        )
    )
    """,
)
_OWNERSHIP_GUARD: Final = threading.Lock()
_OWNERSHIP_PID = os.getpid()
_INITIALIZATION_GUARD: Final = threading.Lock()


@dataclass(slots=True)
class _OwnershipState:
    stream: BinaryIO
    references: int


_OWNERSHIP: Final[dict[Path, _OwnershipState]] = {}


@dataclass(slots=True)
class _InitializationState:
    lock: threading.Lock
    references: int


_INITIALIZATIONS: Final[dict[Path, _InitializationState]] = {}


class DatabaseOwnershipError(RuntimeError):
    """Another process owns a mutable DuckDB database."""


def database_owner_lock_path(path: str | Path) -> Path:
    """Return the canonical cross-process ownership lock for a file database."""

    if str(path) == ":memory:":
        raise ValueError("an in-memory database has no ownership lock path")
    database_path = Path(path).expanduser().resolve()
    return database_path.with_name(f".{database_path.name}.owner.lock")


@contextmanager
def database_ownership(path: str | Path) -> Iterator[Path | str]:
    """Own one mutable DuckDB path across processes, reentrantly per process."""

    raw_path = str(path)
    if raw_path == ":memory:":
        yield raw_path
        return
    database_path = Path(path).expanduser().resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with _OWNERSHIP_GUARD:
        _reset_ownership_after_fork()
        state = _OWNERSHIP.get(database_path)
        if state is None:
            lock_path = database_owner_lock_path(database_path)
            stream = lock_path.open("a+b")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                stream.close()
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise DatabaseOwnershipError(
                        "mutable viewer database is already owned by another "
                        f"process: {database_path}"
                    ) from exc
                raise
            state = _OwnershipState(stream=stream, references=1)
            _OWNERSHIP[database_path] = state
        else:
            state.references += 1
    try:
        yield database_path
    finally:
        with _OWNERSHIP_GUARD:
            state = _OWNERSHIP.get(database_path)
            if state is not None:
                state.references -= 1
                if state.references == 0:
                    del _OWNERSHIP[database_path]
                    try:
                        fcntl.flock(state.stream.fileno(), fcntl.LOCK_UN)
                    finally:
                        state.stream.close()


def _reset_ownership_after_fork() -> None:
    global _OWNERSHIP_PID
    current_pid = os.getpid()
    if current_pid == _OWNERSHIP_PID:
        return
    for state in _OWNERSHIP.values():
        state.stream.close()
    _OWNERSHIP.clear()
    _INITIALIZATIONS.clear()
    _OWNERSHIP_PID = current_pid


@contextmanager
def _serialized_initialization(path: Path | str) -> Iterator[None]:
    if path == ":memory:":
        yield
        return
    database_path = Path(path)
    with _INITIALIZATION_GUARD:
        state = _INITIALIZATIONS.get(database_path)
        if state is None:
            state = _InitializationState(lock=threading.Lock(), references=0)
            _INITIALIZATIONS[database_path] = state
        state.references += 1
    try:
        with state.lock:
            yield
    finally:
        with _INITIALIZATION_GUARD:
            state.references -= 1
            if state.references == 0:
                del _INITIALIZATIONS[database_path]


class ViewerDatabase:
    """Own the one-process DuckDB connection used by the viewer."""

    def __init__(self, path: str | Path) -> None:
        ownership: AbstractContextManager[Path | str] = database_ownership(
            path
        )
        owned_path = ownership.__enter__()
        try:
            with _serialized_initialization(owned_path):
                self._connection = self._open_initialized(str(owned_path))
        except BaseException:
            ownership.__exit__(None, None, None)
            raise
        self._ownership = ownership
        self._closed = False

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._connection

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._connection.close()
        finally:
            self._ownership.__exit__(None, None, None)

    def __enter__(self) -> ViewerDatabase:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def register_runs(self, descriptors: Iterable[RunDescriptor]) -> None:
        """Persist validated provenance without importing Parquet rows."""
        values = tuple(descriptors)
        duplicate_ids = _duplicates(descriptor.run_id for descriptor in values)
        if duplicate_ids:
            raise InvalidQueryError(
                "run IDs must be unique: " + ", ".join(duplicate_ids)
            )
        self._transaction(
            lambda: [self._register_run(descriptor) for descriptor in values]
        )

    def list_tags(self) -> tuple[Tag, ...]:
        rows = self._connection.execute(
            "SELECT tag_id, name FROM tags ORDER BY normalized_name, tag_id"
        ).fetchall()
        return tuple(Tag(tag_id=row[0], name=row[1]) for row in rows)

    def create_tag(self, name: str) -> Tag:
        normalized_name, display_name = _normalize_tag_name(name)
        existing = self._connection.execute(
            "SELECT tag_id, name FROM tags WHERE normalized_name = ?",
            [normalized_name],
        ).fetchone()
        if existing is not None:
            return Tag(tag_id=existing[0], name=existing[1])
        tag = Tag(tag_id=uuid.uuid4().hex, name=display_name)

        def insert() -> None:
            self._connection.execute(
                """
                INSERT INTO tags(tag_id, name, normalized_name)
                VALUES (?, ?, ?)
                """,
                [tag.tag_id, tag.name, normalized_name],
            )

        try:
            self._transaction(insert)
        except duckdb.ConstraintException:
            # A future threaded adapter can race on normalized_name safely.
            row = self._connection.execute(
                "SELECT tag_id, name FROM tags WHERE normalized_name = ?",
                [normalized_name],
            ).fetchone()
            if row is None:
                raise
            return Tag(tag_id=row[0], name=row[1])
        return tag

    def get_annotation(
        self,
        corpus_sha256: str,
        sample_id: str,
        decoder_output_sha256: str,
    ) -> Annotation | None:
        corpus_sha256, sample_id, decoder_output_sha256 = _annotation_key(
            corpus_sha256, sample_id, decoder_output_sha256
        )
        row = self._connection.execute(
            """
            SELECT verdict, note
            FROM annotations
            WHERE corpus_sha256 = ? AND sample_id = ?
              AND decoder_output_sha256 = ?
            """,
            [corpus_sha256, sample_id, decoder_output_sha256],
        ).fetchone()
        if row is None:
            return None
        tags = self._annotation_tags(
            corpus_sha256, sample_id, decoder_output_sha256
        )
        return Annotation(
            corpus_sha256=corpus_sha256,
            sample_id=sample_id,
            decoder_output_sha256=decoder_output_sha256,
            verdict=Verdict(row[0]) if row[0] is not None else None,
            note=row[1],
            tags=tags,
        )

    def get_annotations(
        self,
        corpus_sha256: str,
        identities: Iterable[tuple[str, str]],
    ) -> dict[tuple[str, str], Annotation]:
        """Load annotations and their tags for exact output identities."""
        requested = tuple(
            sorted(
                {
                    _annotation_key(corpus_sha256, sample_id, output_sha256)[
                        1:
                    ]
                    for sample_id, output_sha256 in identities
                }
            )
        )
        if not requested:
            return {}
        values_sql = ", ".join("(?, ?)" for _ in requested)
        params = [item for identity in requested for item in identity]
        rows = self._connection.execute(
            f"""
            WITH requested(sample_id, decoder_output_sha256) AS (
                VALUES {values_sql}
            )
            SELECT
                a.sample_id,
                a.decoder_output_sha256,
                a.verdict,
                a.note,
                t.tag_id,
                t.name
            FROM requested AS requested
            JOIN annotations AS a
              ON a.corpus_sha256 = ?
             AND a.sample_id = requested.sample_id
             AND a.decoder_output_sha256 = requested.decoder_output_sha256
            LEFT JOIN annotation_tags AS atag
              ON atag.corpus_sha256 = a.corpus_sha256
             AND atag.sample_id = a.sample_id
             AND atag.decoder_output_sha256 = a.decoder_output_sha256
            LEFT JOIN tags AS t ON t.tag_id = atag.tag_id
            ORDER BY
                a.sample_id,
                a.decoder_output_sha256,
                t.normalized_name,
                t.tag_id
            """,
            [*params, corpus_sha256],
        ).fetchall()
        grouped: dict[
            tuple[str, str], tuple[Verdict | None, str | None, list[Tag]]
        ] = {}
        for row in rows:
            key = (row[0], row[1])
            state = grouped.setdefault(
                key,
                (
                    Verdict(row[2]) if row[2] is not None else None,
                    row[3],
                    [],
                ),
            )
            if row[4] is not None:
                state[2].append(Tag(tag_id=row[4], name=row[5]))
        return {
            key: Annotation(
                corpus_sha256=corpus_sha256,
                sample_id=key[0],
                decoder_output_sha256=key[1],
                verdict=state[0],
                note=state[1],
                tags=tuple(state[2]),
            )
            for key, state in grouped.items()
        }

    def put_annotation(
        self,
        corpus_sha256: str,
        sample_id: str,
        decoder_output_sha256: str,
        *,
        verdict: Verdict | str | None,
        note: str | None,
        tag_ids: Iterable[str] = (),
    ) -> Annotation:
        corpus_sha256, sample_id, decoder_output_sha256 = _annotation_key(
            corpus_sha256, sample_id, decoder_output_sha256
        )
        if verdict is None:
            parsed_verdict = None
        else:
            try:
                parsed_verdict = Verdict(verdict)
            except ValueError as exc:
                raise InvalidQueryError(
                    f"unsupported annotation verdict: {verdict}"
                ) from exc
        normalized_note = _normalize_note(note)
        selected_tag_ids = tuple(sorted(set(tag_ids)))
        if any(
            not isinstance(tag_id, str) or not tag_id
            for tag_id in selected_tag_ids
        ):
            raise InvalidQueryError("tag_ids must contain nonblank strings")

        def upsert() -> None:
            if selected_tag_ids:
                placeholders = ", ".join("?" for _ in selected_tag_ids)
                found = {
                    row[0]
                    for row in self._connection.execute(
                        f"SELECT tag_id FROM tags WHERE tag_id IN ({placeholders})",
                        list(selected_tag_ids),
                    ).fetchall()
                }
                missing = sorted(set(selected_tag_ids).difference(found))
                if missing:
                    raise InvalidQueryError(
                        "unknown tag ID(s): " + ", ".join(missing)
                    )
            self._connection.execute(
                """
                INSERT INTO annotations(
                    corpus_sha256, sample_id, decoder_output_sha256,
                    verdict, note
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(
                    corpus_sha256, sample_id, decoder_output_sha256
                ) DO UPDATE SET
                    verdict = excluded.verdict,
                    note = excluded.note,
                    updated_at = now()
                """,
                [
                    corpus_sha256,
                    sample_id,
                    decoder_output_sha256,
                    parsed_verdict.value
                    if parsed_verdict is not None
                    else None,
                    normalized_note,
                ],
            )
            self._connection.execute(
                """
                DELETE FROM annotation_tags
                WHERE corpus_sha256 = ? AND sample_id = ?
                  AND decoder_output_sha256 = ?
                """,
                [corpus_sha256, sample_id, decoder_output_sha256],
            )
            for tag_id in selected_tag_ids:
                self._connection.execute(
                    """
                    INSERT INTO annotation_tags(
                        corpus_sha256, sample_id, decoder_output_sha256, tag_id
                    ) VALUES (?, ?, ?, ?)
                    """,
                    [corpus_sha256, sample_id, decoder_output_sha256, tag_id],
                )

        self._transaction(upsert)
        annotation = self.get_annotation(
            corpus_sha256, sample_id, decoder_output_sha256
        )
        assert annotation is not None
        return annotation

    def delete_annotation(
        self,
        corpus_sha256: str,
        sample_id: str,
        decoder_output_sha256: str,
    ) -> bool:
        corpus_sha256, sample_id, decoder_output_sha256 = _annotation_key(
            corpus_sha256, sample_id, decoder_output_sha256
        )
        deleted = False

        def remove() -> None:
            nonlocal deleted
            self._connection.execute(
                """
                DELETE FROM annotation_tags
                WHERE corpus_sha256 = ? AND sample_id = ?
                  AND decoder_output_sha256 = ?
                """,
                [corpus_sha256, sample_id, decoder_output_sha256],
            )
            result = self._connection.execute(
                """
                DELETE FROM annotations
                WHERE corpus_sha256 = ? AND sample_id = ?
                  AND decoder_output_sha256 = ?
                RETURNING sample_id
                """,
                [corpus_sha256, sample_id, decoder_output_sha256],
            ).fetchone()
            deleted = result is not None

        self._transaction(remove)
        return deleted

    def export_annotations(self) -> list[dict[str, object]]:
        """Return a timestamp- and machine-path-free deterministic export."""
        rows = self._connection.execute(
            """
            SELECT
                a.corpus_sha256,
                a.sample_id,
                a.decoder_output_sha256,
                a.verdict,
                a.note,
                coalesce(
                    list(t.name ORDER BY t.normalized_name, t.tag_id)
                        FILTER (WHERE t.tag_id IS NOT NULL),
                    []
                ) AS tags
            FROM annotations AS a
            LEFT JOIN annotation_tags AS atag USING (
                corpus_sha256, sample_id, decoder_output_sha256
            )
            LEFT JOIN tags AS t USING (tag_id)
            GROUP BY
                a.corpus_sha256,
                a.sample_id,
                a.decoder_output_sha256,
                a.verdict,
                a.note
            ORDER BY
                a.corpus_sha256,
                a.sample_id,
                a.decoder_output_sha256
            """
        ).fetchall()
        return [
            {
                "corpus_sha256": row[0],
                "sample_id": row[1],
                "decoder_output_sha256": row[2],
                "verdict": row[3],
                "note": row[4],
                "tags": row[5],
            }
            for row in rows
        ]

    def _annotation_tags(
        self, corpus_sha256: str, sample_id: str, decoder_output_sha256: str
    ) -> tuple[Tag, ...]:
        rows = self._connection.execute(
            """
            SELECT t.tag_id, t.name
            FROM annotation_tags AS atag
            JOIN tags AS t USING (tag_id)
            WHERE atag.corpus_sha256 = ? AND atag.sample_id = ?
              AND atag.decoder_output_sha256 = ?
            ORDER BY t.normalized_name, t.tag_id
            """,
            [corpus_sha256, sample_id, decoder_output_sha256],
        ).fetchall()
        return tuple(Tag(tag_id=row[0], name=row[1]) for row in rows)

    def _register_run(self, descriptor: RunDescriptor) -> None:
        self._connection.execute(
            """
            INSERT INTO runs(
                run_id, label, descriptor_json, manifest_sha256,
                corpus_sha256, definition_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                label = excluded.label,
                descriptor_json = excluded.descriptor_json,
                manifest_sha256 = excluded.manifest_sha256,
                corpus_sha256 = excluded.corpus_sha256,
                definition_id = excluded.definition_id,
                registered_at = now()
            """,
            [
                descriptor.run_id,
                descriptor.label,
                descriptor.to_json(),
                descriptor.preprocessing_manifest_sha256,
                descriptor.corpus_sha256,
                descriptor.definition_id,
            ],
        )

    @staticmethod
    def _open_initialized(
        raw_path: str,
    ) -> duckdb.DuckDBPyConnection:
        connection = duckdb.connect(raw_path)
        try:
            _create_schema(connection)
        except BaseException:
            connection.close()
            raise
        return connection

    def _transaction(self, operation: Callable[[], object]) -> None:
        self._connection.execute("BEGIN TRANSACTION")
        try:
            operation()
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        self._connection.execute("COMMIT")


def _annotation_key(
    corpus_sha256: str, sample_id: str, decoder_output_sha256: str
) -> tuple[str, str, str]:
    validate_sha256(corpus_sha256, "corpus_sha256")
    validate_sha256(decoder_output_sha256, "decoder_output_sha256")
    if not isinstance(sample_id, str) or not sample_id:
        raise InvalidQueryError("sample_id must not be blank")
    return corpus_sha256, sample_id, decoder_output_sha256


def _normalize_tag_name(name: str) -> tuple[str, str]:
    if not isinstance(name, str):
        raise InvalidQueryError("tag name must be a string")
    display_name = " ".join(name.split())
    if not display_name:
        raise InvalidQueryError("tag name must not be blank")
    if len(display_name) > 100:
        raise InvalidQueryError("tag name must be at most 100 characters")
    return display_name.casefold(), display_name


def _normalize_note(note: str | None) -> str | None:
    if note is None:
        return None
    if not isinstance(note, str):
        raise InvalidQueryError("annotation note must be a string or null")
    if len(note) > 20_000:
        raise InvalidQueryError(
            "annotation note must be at most 20000 characters"
        )
    return note


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _create_schema(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute("BEGIN TRANSACTION")
    try:
        for statement in _SCHEMA:
            connection.execute(statement)
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")


__all__ = (
    "DatabaseOwnershipError",
    "ViewerDatabase",
    "database_owner_lock_path",
    "database_ownership",
)
