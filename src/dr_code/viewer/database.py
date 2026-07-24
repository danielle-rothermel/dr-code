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
from enum import UNIQUE, StrEnum, verify
from functools import cache
from pathlib import Path
from typing import BinaryIO, Final

import duckdb

from dr_code.viewer.domain import (
    ANNOTATION_TAG_IDS_MAX_COUNT,
    Annotation,
    InvalidQueryError,
    RunDescriptor,
    Tag,
    Verdict,
    normalize_annotation_tag_ids,
    normalize_tag_name,
    validate_annotation_note,
    validate_sha256,
)


_LEGACY_SCHEMA = (
    """
    CREATE TABLE runs (
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
        PRIMARY KEY (corpus_sha256, sample_id, decoder_output_sha256)
    )
    """,
    """
    CREATE TABLE tags (
        tag_id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        normalized_name VARCHAR NOT NULL UNIQUE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
    )
    """,
    """
    CREATE TABLE annotation_tags (
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
# Persisted archive format: exact table and column literals are schema-tested.
_ARCHIVE_SCHEMA = (
    """
    CREATE TABLE archived_annotations (
        corpus_sha256 VARCHAR NOT NULL,
        sample_id VARCHAR NOT NULL,
        decoder_output_sha256 VARCHAR NOT NULL,
        verdict VARCHAR,
        note VARCHAR,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        archive_reason VARCHAR NOT NULL,
        source_schema_version INTEGER NOT NULL,
        PRIMARY KEY (corpus_sha256, sample_id, decoder_output_sha256)
    )
    """,
    """
    CREATE TABLE archived_annotation_tags (
        corpus_sha256 VARCHAR NOT NULL,
        sample_id VARCHAR NOT NULL,
        decoder_output_sha256 VARCHAR NOT NULL,
        tag_id VARCHAR NOT NULL,
        tag_name VARCHAR NOT NULL,
        normalized_tag_name VARCHAR NOT NULL,
        archive_reason VARCHAR NOT NULL,
        source_schema_version INTEGER NOT NULL,
        PRIMARY KEY (
            corpus_sha256, sample_id, decoder_output_sha256, tag_id
        )
    )
    """,
    """
    CREATE TABLE archived_tags (
        tag_id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        normalized_name VARCHAR NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        archive_reason VARCHAR NOT NULL,
        source_schema_version INTEGER NOT NULL
    )
    """,
)
_SCHEMA = (*_LEGACY_SCHEMA, *_ARCHIVE_SCHEMA)
_LEGACY_SCHEMA_TABLES: Final = frozenset(
    {"runs", "annotations", "tags", "annotation_tags"}
)
_SCHEMA_TABLES: Final = _LEGACY_SCHEMA_TABLES | {
    "archived_annotations",
    "archived_annotation_tags",
    "archived_tags",
}
_LEGACY_SCHEMA_VERSION: Final = 1
_SCHEMA_VERSION: Final = 2
_OWNERSHIP_GUARD = threading.Lock()
_INITIALIZATION_GUARD = threading.Lock()


@verify(UNIQUE)
class _ArchiveReason(StrEnum):
    """Persisted reason literals; never iterate to build stored payloads."""

    NOTE_EXCEEDS_MAX_LENGTH = "note_exceeds_max_length"
    NOTE_IS_NOT_UNICODE_SCALAR = "note_is_not_unicode_scalar"
    NORMALIZED_TAG_NAME_IS_NOT_CANONICAL = (
        "normalized_tag_name_is_not_canonical"
    )
    REFERENCES_ARCHIVED_TAG = "references_archived_tag"
    TAG_COUNT_EXCEEDS_MAXIMUM = "tag_count_exceeds_maximum"
    TAG_NAME_IS_MALFORMED = "tag_name_is_malformed"
    TAG_NAME_IS_NOT_NORMALIZED = "tag_name_is_not_normalized"
    TAG_NAME_IS_OUT_OF_CONTRACT = "tag_name_is_out_of_contract"


@verify(UNIQUE)
class _CatalogObjectKind(StrEnum):
    FUNCTION = "function"
    INDEX = "index"
    SCHEMA = "schema"
    SEQUENCE = "sequence"
    TABLE = "table"
    TYPE = "type"
    VIEW = "view"


@dataclass(frozen=True, slots=True)
class _CatalogObject:
    kind: _CatalogObjectKind
    database_name: str
    schema_name: str
    name: str


@dataclass(slots=True)
class _OwnershipState:
    stream: BinaryIO
    references: int
    acquisition_pid: int


_OWNERSHIP: Final[dict[Path, _OwnershipState]] = {}


@dataclass(slots=True)
class _InitializationState:
    lock: threading.Lock
    references: int


_INITIALIZATIONS: Final[dict[Path, _InitializationState]] = {}


def _reset_process_state_after_fork() -> None:
    """Discard inherited synchronization state without unlocking parent locks."""

    global _INITIALIZATION_GUARD, _OWNERSHIP_GUARD
    for state in _OWNERSHIP.values():
        state.stream.close()
    _OWNERSHIP.clear()
    _INITIALIZATIONS.clear()
    _OWNERSHIP_GUARD = threading.Lock()
    _INITIALIZATION_GUARD = threading.Lock()


os.register_at_fork(after_in_child=_reset_process_state_after_fork)


class DatabaseOwnershipError(RuntimeError):
    """Another process owns a mutable DuckDB database."""


class DatabaseSchemaError(RuntimeError):
    """The database cannot be opened without ambiguous persisted state."""


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
    acquisition_pid = os.getpid()
    with _OWNERSHIP_GUARD:
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
            state = _OwnershipState(
                stream=stream,
                references=1,
                acquisition_pid=acquisition_pid,
            )
            _OWNERSHIP[database_path] = state
        else:
            state.references += 1
    try:
        yield database_path
    finally:
        if os.getpid() != acquisition_pid:
            return
        with _OWNERSHIP_GUARD:
            state = _OWNERSHIP.get(database_path)
            if state is not None and state.acquisition_pid == acquisition_pid:
                state.references -= 1
                if state.references == 0:
                    del _OWNERSHIP[database_path]
                    try:
                        fcntl.flock(state.stream.fileno(), fcntl.LOCK_UN)
                    finally:
                        state.stream.close()


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
        self._creator_pid = os.getpid()

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        current_pid = os.getpid()
        if current_pid != self._creator_pid:
            raise DatabaseOwnershipError(
                "viewer database connection belongs to process "
                f"{self._creator_pid} and cannot be used in process "
                f"{current_pid}"
            )
        return self._connection

    def close(self) -> None:
        if os.getpid() != self._creator_pid:
            return
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
        rows = self.connection.execute(
            "SELECT tag_id, name FROM tags ORDER BY normalized_name, tag_id"
        ).fetchall()
        return tuple(Tag(tag_id=row[0], name=row[1]) for row in rows)

    def create_tag(self, name: str) -> Tag:
        normalized_name, display_name = normalize_tag_name(name)
        existing = self.connection.execute(
            "SELECT tag_id, name FROM tags WHERE normalized_name = ?",
            [normalized_name],
        ).fetchone()
        if existing is not None:
            return Tag(tag_id=existing[0], name=existing[1])
        tag = Tag(tag_id=uuid.uuid4().hex, name=display_name)

        def insert() -> None:
            self.connection.execute(
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
            row = self.connection.execute(
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
        row = self.connection.execute(
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
        rows = self.connection.execute(
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
        normalized_note = validate_annotation_note(note)
        selected_tag_ids = normalize_annotation_tag_ids(tag_ids)

        def upsert() -> None:
            if selected_tag_ids:
                placeholders = ", ".join("?" for _ in selected_tag_ids)
                found = {
                    row[0]
                    for row in self.connection.execute(
                        f"SELECT tag_id FROM tags WHERE tag_id IN ({placeholders})",
                        list(selected_tag_ids),
                    ).fetchall()
                }
                missing = sorted(set(selected_tag_ids).difference(found))
                if missing:
                    raise InvalidQueryError(
                        "unknown tag ID(s): " + ", ".join(missing)
                    )
            self.connection.execute(
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
            self.connection.execute(
                """
                DELETE FROM annotation_tags
                WHERE corpus_sha256 = ? AND sample_id = ?
                  AND decoder_output_sha256 = ?
                """,
                [corpus_sha256, sample_id, decoder_output_sha256],
            )
            for tag_id in selected_tag_ids:
                self.connection.execute(
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
            self.connection.execute(
                """
                DELETE FROM annotation_tags
                WHERE corpus_sha256 = ? AND sample_id = ?
                  AND decoder_output_sha256 = ?
                """,
                [corpus_sha256, sample_id, decoder_output_sha256],
            )
            result = self.connection.execute(
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
        rows = self.connection.execute(
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
        rows = self.connection.execute(
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
        self.connection.execute(
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
        self.connection.execute("BEGIN TRANSACTION")
        try:
            operation()
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        self.connection.execute("COMMIT")


def _annotation_key(
    corpus_sha256: str, sample_id: str, decoder_output_sha256: str
) -> tuple[str, str, str]:
    validate_sha256(corpus_sha256, "corpus_sha256")
    validate_sha256(decoder_output_sha256, "decoder_output_sha256")
    if not isinstance(sample_id, str) or not sample_id:
        raise InvalidQueryError("sample_id must not be blank")
    return corpus_sha256, sample_id, decoder_output_sha256


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
        _admit_or_initialize_schema(connection)
        _validate_current_schema(connection)
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")


def _admit_or_initialize_schema(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    catalog_objects = _user_catalog_objects(connection)
    tables = _main_tables(connection)
    unsupported_objects = tuple(
        sorted(
            (
                catalog_object
                for catalog_object in catalog_objects
                if catalog_object.kind is not _CatalogObjectKind.TABLE
                or catalog_object.schema_name != "main"
            ),
            key=lambda catalog_object: (
                catalog_object.kind,
                catalog_object.database_name,
                catalog_object.schema_name,
                catalog_object.name,
            ),
        )
    )
    if unsupported_objects:
        rendered = ", ".join(
            f"{catalog_object.kind} "
            f"{catalog_object.database_name}."
            f"{catalog_object.schema_name}.{catalog_object.name}"
            for catalog_object in unsupported_objects
        )
        raise DatabaseSchemaError(
            "viewer database catalog contains unsupported persistent user "
            f"objects: {rendered}"
        )

    if "viewer_schema" in tables:
        _validate_table_schema(
            connection, "viewer_schema", _LEGACY_SCHEMA_VERSION
        )
        versions = connection.execute(
            "SELECT schema_version FROM viewer_schema"
        ).fetchall()
        if versions == [(_SCHEMA_VERSION,)]:
            _validate_schema_tables(
                connection,
                tables,
                _SCHEMA_TABLES | {"viewer_schema"},
                _SCHEMA_VERSION,
            )
            _archive_out_of_contract_rows(connection, _SCHEMA_VERSION)
            return
        if versions != [(_LEGACY_SCHEMA_VERSION,)]:
            raise DatabaseSchemaError(
                "viewer database schema version is unsupported: "
                f"expected {_LEGACY_SCHEMA_VERSION} or {_SCHEMA_VERSION}, "
                f"found {versions!r}"
            )
        _validate_schema_tables(
            connection,
            tables,
            _LEGACY_SCHEMA_TABLES | {"viewer_schema"},
            _LEGACY_SCHEMA_VERSION,
        )
        _migrate_legacy_annotation_contract(connection)
        return

    if not catalog_objects:
        for statement in _SCHEMA:
            connection.execute(statement)
        initial_version = _SCHEMA_VERSION
    elif tables == _LEGACY_SCHEMA_TABLES:
        # This is the sole supported unversioned legacy signature. Prove the
        # entire shape before adding version metadata.
        _validate_schema_tables(
            connection,
            tables,
            _LEGACY_SCHEMA_TABLES,
            _LEGACY_SCHEMA_VERSION,
        )
        initial_version = _LEGACY_SCHEMA_VERSION
    else:
        missing = sorted(_LEGACY_SCHEMA_TABLES.difference(tables))
        unexpected = sorted(tables.difference(_LEGACY_SCHEMA_TABLES))
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise DatabaseSchemaError(
            "viewer database unversioned schema is unsupported: "
            + "; ".join(details)
        )

    connection.execute(
        """
        CREATE TABLE viewer_schema (
            schema_version INTEGER PRIMARY KEY
        )
        """
    )
    connection.execute(
        "INSERT INTO viewer_schema(schema_version) VALUES (?)",
        [initial_version],
    )
    if initial_version == _LEGACY_SCHEMA_VERSION:
        _migrate_legacy_annotation_contract(connection)


def _migrate_legacy_annotation_contract(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    for statement in _ARCHIVE_SCHEMA:
        connection.execute(statement)
    _archive_out_of_contract_rows(connection, _LEGACY_SCHEMA_VERSION)
    connection.execute(
        "UPDATE viewer_schema SET schema_version = ?",
        [_SCHEMA_VERSION],
    )


def _archive_out_of_contract_rows(
    connection: duckdb.DuckDBPyConnection, source_schema_version: int
) -> None:
    tag_rows = connection.execute(
        """
        SELECT tag_id, name, normalized_name
        FROM tags
        ORDER BY tag_id
        """
    ).fetchall()
    invalid_tags: dict[str, str] = {}
    for row in tag_rows:
        reasons = _tag_archive_reasons(row[1], row[2])
        if reasons:
            invalid_tags[row[0]] = ",".join(reasons)

    annotation_rows = connection.execute(
        """
        SELECT
            corpus_sha256,
            sample_id,
            decoder_output_sha256,
            verdict,
            note
        FROM annotations
        ORDER BY corpus_sha256, sample_id, decoder_output_sha256
        """
    ).fetchall()
    tag_links_by_annotation: dict[
        tuple[object, ...], list[tuple[object, ...]]
    ] = {}
    for row in connection.execute(
        """
        SELECT
            atag.corpus_sha256,
            atag.sample_id,
            atag.decoder_output_sha256,
            atag.tag_id,
            tag.name,
            tag.normalized_name
        FROM annotation_tags AS atag
        JOIN tags AS tag USING (tag_id)
        ORDER BY
            atag.corpus_sha256,
            atag.sample_id,
            atag.decoder_output_sha256,
            atag.tag_id
        """
    ).fetchall():
        tag_links_by_annotation.setdefault(tuple(row[:3]), []).append(
            tuple(row[3:])
        )
    annotations_to_archive: list[
        tuple[tuple[object, ...], tuple[tuple[object, ...], ...], str]
    ] = []
    for annotation_row in annotation_rows:
        identity = tuple(annotation_row[:3])
        tag_links = tuple(tag_links_by_annotation.get(identity, ()))
        reasons: list[_ArchiveReason] = []
        note_reason = _annotation_note_archive_reason(annotation_row[4])
        if note_reason is not None:
            reasons.append(note_reason)
        if len(tag_links) > ANNOTATION_TAG_IDS_MAX_COUNT:
            reasons.append(_ArchiveReason.TAG_COUNT_EXCEEDS_MAXIMUM)
        if any(tag_id in invalid_tags for tag_id, *_rest in tag_links):
            reasons.append(_ArchiveReason.REFERENCES_ARCHIVED_TAG)
        if reasons:
            annotations_to_archive.append(
                (tuple(annotation_row), tag_links, ",".join(reasons))
            )

    for annotation_row, tag_links, reason in annotations_to_archive:
        identity = annotation_row[:3]
        if (
            connection.execute(
                """
            SELECT 1
            FROM archived_annotations
            WHERE corpus_sha256 = ? AND sample_id = ?
              AND decoder_output_sha256 = ?
            """,
                list(identity),
            ).fetchone()
            is not None
        ):
            raise DatabaseSchemaError(
                "active out-of-contract annotation duplicates an archived "
                f"identity: {identity!r}; export the archive and resolve the "
                "duplicate before reopening the viewer database"
            )
        connection.execute(
            """
            INSERT INTO archived_annotations(
                corpus_sha256,
                sample_id,
                decoder_output_sha256,
                verdict,
                note,
                created_at,
                updated_at,
                archive_reason,
                source_schema_version
            )
            SELECT
                corpus_sha256,
                sample_id,
                decoder_output_sha256,
                verdict,
                note,
                created_at,
                updated_at,
                ?,
                ?
            FROM annotations
            WHERE corpus_sha256 = ? AND sample_id = ?
              AND decoder_output_sha256 = ?
            """,
            [reason, source_schema_version, *identity],
        )
        for tag_id, tag_name, normalized_tag_name in tag_links:
            connection.execute(
                """
                INSERT INTO archived_annotation_tags(
                    corpus_sha256,
                    sample_id,
                    decoder_output_sha256,
                    tag_id,
                    tag_name,
                    normalized_tag_name,
                    archive_reason,
                    source_schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    *identity,
                    tag_id,
                    tag_name,
                    normalized_tag_name,
                    reason,
                    source_schema_version,
                ],
            )
        connection.execute(
            """
            DELETE FROM annotation_tags
            WHERE corpus_sha256 = ? AND sample_id = ?
              AND decoder_output_sha256 = ?
            """,
            list(identity),
        )
        connection.execute(
            """
            DELETE FROM annotations
            WHERE corpus_sha256 = ? AND sample_id = ?
              AND decoder_output_sha256 = ?
            """,
            list(identity),
        )

    for tag_id, reason in invalid_tags.items():
        if (
            connection.execute(
                "SELECT 1 FROM archived_tags WHERE tag_id = ?", [tag_id]
            ).fetchone()
            is not None
        ):
            raise DatabaseSchemaError(
                "active out-of-contract tag duplicates an archived ID: "
                f"{tag_id!r}; export the archive and resolve the duplicate "
                "before reopening the viewer database"
            )
        connection.execute(
            """
            INSERT INTO archived_tags(
                tag_id,
                name,
                normalized_name,
                created_at,
                archive_reason,
                source_schema_version
            )
            SELECT
                tag_id,
                name,
                normalized_name,
                created_at,
                ?,
                ?
            FROM tags
            WHERE tag_id = ?
            """,
            [reason, source_schema_version, tag_id],
        )
    if invalid_tags:
        placeholders = ", ".join("?" for _tag_id in invalid_tags)
        remaining_links = connection.execute(
            f"""
            SELECT tag_id
            FROM annotation_tags
            WHERE tag_id IN ({placeholders})
            ORDER BY tag_id
            """,
            list(invalid_tags),
        ).fetchall()
        if remaining_links:
            raise DatabaseSchemaError(
                "annotation archive migration left references to archived "
                f"tags: {remaining_links!r}"
            )
        # DuckDB does not permit parent-row deletion after child-row deletion
        # in one transaction. Rebuild the exact link table after proving the
        # closure above, preserving every remaining active relationship.
        connection.execute(
            "CREATE TEMP TABLE retained_annotation_tags AS "
            "SELECT * FROM annotation_tags"
        )
        connection.execute("DROP TABLE annotation_tags")
        connection.execute(
            f"DELETE FROM tags WHERE tag_id IN ({placeholders})",
            list(invalid_tags),
        )
        connection.execute(_LEGACY_SCHEMA[3])
        connection.execute(
            "INSERT INTO annotation_tags SELECT * FROM retained_annotation_tags"
        )
        connection.execute("DROP TABLE retained_annotation_tags")


def _tag_archive_reasons(
    name: object, normalized_name: object
) -> tuple[_ArchiveReason, ...]:
    if not isinstance(name, str) or not isinstance(normalized_name, str):
        return (_ArchiveReason.TAG_NAME_IS_MALFORMED,)
    try:
        canonical_normalized_name, display_name = normalize_tag_name(name)
    except InvalidQueryError:
        return (_ArchiveReason.TAG_NAME_IS_OUT_OF_CONTRACT,)
    reasons: list[_ArchiveReason] = []
    if name != display_name:
        reasons.append(_ArchiveReason.TAG_NAME_IS_NOT_NORMALIZED)
    if normalized_name != canonical_normalized_name:
        reasons.append(_ArchiveReason.NORMALIZED_TAG_NAME_IS_NOT_CANONICAL)
    return tuple(reasons)


def _annotation_note_archive_reason(
    note: str | None,
) -> _ArchiveReason | None:
    try:
        validate_annotation_note(note)
    except InvalidQueryError:
        if isinstance(note, str):
            try:
                note.encode("utf-8")
            except UnicodeEncodeError:
                return _ArchiveReason.NOTE_IS_NOT_UNICODE_SCALAR
        return _ArchiveReason.NOTE_EXCEEDS_MAX_LENGTH
    return None


def _validate_current_schema(connection: duckdb.DuckDBPyConnection) -> None:
    tables = _main_tables(connection)
    expected_tables = _SCHEMA_TABLES.union({"viewer_schema"})
    _validate_schema_tables(
        connection, tables, expected_tables, _SCHEMA_VERSION
    )
    versions = connection.execute(
        "SELECT schema_version FROM viewer_schema"
    ).fetchall()
    if versions != [(_SCHEMA_VERSION,)]:
        raise DatabaseSchemaError(
            "viewer database schema version is unsupported: "
            f"expected {_SCHEMA_VERSION}, found {versions!r}"
        )


def _main_tables(connection: duckdb.DuckDBPyConnection) -> frozenset[str]:
    return frozenset(
        row[0]
        for row in connection.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            """
        ).fetchall()
    )


def _user_catalog_objects(
    connection: duckdb.DuckDBPyConnection,
) -> frozenset[_CatalogObject]:
    rows = connection.execute(
        """
        SELECT
            'table' AS object_kind,
            database_name,
            schema_name,
            table_name AS object_name
        FROM duckdb_tables()
        WHERE NOT internal AND NOT temporary
        UNION ALL
        SELECT
            'view' AS object_kind,
            database_name,
            schema_name,
            view_name AS object_name
        FROM duckdb_views()
        WHERE NOT internal AND NOT temporary
        UNION ALL
        SELECT
            'sequence' AS object_kind,
            catalog_sequence.database_name,
            catalog_sequence.schema_name,
            catalog_sequence.sequence_name AS object_name
        FROM duckdb_sequences() AS catalog_sequence
        JOIN duckdb_databases() AS catalog_database
          ON catalog_database.database_name = catalog_sequence.database_name
        WHERE NOT catalog_sequence.temporary AND NOT catalog_database.internal
        UNION ALL
        SELECT
            'schema' AS object_kind,
            database_name,
            schema_name,
            schema_name AS object_name
        FROM duckdb_schemas()
        WHERE NOT internal
        UNION ALL
        SELECT
            'type' AS object_kind,
            catalog_type.database_name,
            catalog_type.schema_name,
            catalog_type.type_name AS object_name
        FROM duckdb_types() AS catalog_type
        JOIN duckdb_databases() AS catalog_database
          ON catalog_database.database_name = catalog_type.database_name
        WHERE NOT catalog_type.internal AND NOT catalog_database.internal
        UNION ALL
        SELECT
            'function' AS object_kind,
            catalog_function.database_name,
            catalog_function.schema_name,
            catalog_function.function_name AS object_name
        FROM duckdb_functions() AS catalog_function
        JOIN duckdb_databases() AS catalog_database
          ON catalog_database.database_name = catalog_function.database_name
        WHERE NOT catalog_function.internal AND NOT catalog_database.internal
        UNION ALL
        SELECT
            'index' AS object_kind,
            catalog_index.database_name,
            catalog_index.schema_name,
            catalog_index.index_name AS object_name
        FROM duckdb_indexes() AS catalog_index
        JOIN duckdb_databases() AS catalog_database
          ON catalog_database.database_name = catalog_index.database_name
        WHERE NOT catalog_database.internal
        """
    ).fetchall()
    return frozenset(
        _CatalogObject(
            kind=_CatalogObjectKind(row[0]),
            database_name=row[1],
            schema_name=row[2],
            name=row[3],
        )
        for row in rows
    )


def _validate_schema_tables(
    connection: duckdb.DuckDBPyConnection,
    tables: frozenset[str],
    expected_tables: frozenset[str],
    schema_version: int = _SCHEMA_VERSION,
) -> None:
    if tables != expected_tables:
        missing = sorted(expected_tables.difference(tables))
        unexpected = sorted(tables.difference(expected_tables))
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise DatabaseSchemaError(
            "viewer database schema is incomplete: " + "; ".join(details)
        )
    for table in sorted(expected_tables):
        _validate_table_schema(connection, table, schema_version)


def _validate_table_schema(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    schema_version: int = _SCHEMA_VERSION,
) -> None:
    signature = _table_schema_signature(connection, table)
    expected = _expected_table_schema_signature(table, schema_version)
    if signature != expected:
        raise DatabaseSchemaError(
            f"viewer database schema for {table} is malformed: "
            f"expected {expected!r}, found {signature!r}"
        )


@cache
def _expected_table_schema_signature(
    table: str, schema_version: int = _SCHEMA_VERSION
) -> tuple[tuple[tuple[object, ...], ...], tuple[tuple[object, ...], ...]]:
    with duckdb.connect(":memory:") as expected:
        schema = (
            _LEGACY_SCHEMA
            if schema_version == _LEGACY_SCHEMA_VERSION
            else _SCHEMA
        )
        for statement in schema:
            expected.execute(statement)
        expected.execute(
            "CREATE TABLE viewer_schema (schema_version INTEGER PRIMARY KEY)"
        )
        return _table_schema_signature(expected, table)


def _table_schema_signature(
    connection: duckdb.DuckDBPyConnection,
    table: str,
) -> tuple[tuple[tuple[object, ...], ...], tuple[tuple[object, ...], ...]]:
    columns = tuple(
        tuple(row[1:])
        for row in connection.execute(
            f"PRAGMA table_info('{table}')"
        ).fetchall()
    )
    raw_constraints = connection.execute(
        """
        SELECT
            constraint_type,
            constraint_text,
            expression,
            constraint_column_names,
            referenced_table,
            referenced_column_names
        FROM duckdb_constraints()
        WHERE table_name = ?
        ORDER BY constraint_type, constraint_text
        """,
        [table],
    ).fetchall()
    constraints = tuple(
        tuple(
            tuple(value) if isinstance(value, list) else value for value in row
        )
        for row in raw_constraints
    )
    return columns, constraints


__all__ = (
    "DatabaseOwnershipError",
    "DatabaseSchemaError",
    "ViewerDatabase",
    "database_owner_lock_path",
    "database_ownership",
)
