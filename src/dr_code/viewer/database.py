"""DuckDB run catalog and example-annotation persistence."""

from __future__ import annotations

import errno
import fcntl
import os
import threading
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from enum import UNIQUE, StrEnum, verify
from functools import cache
from pathlib import Path
from typing import BinaryIO, Final, cast

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from dr_code.corpus.stable_files import stable_files
from dr_code.viewer.domain import (
    ANNOTATION_TAG_IDS_MAX_COUNT,
    Annotation,
    InvalidQueryError,
    InvalidTaskAnnotationError,
    MachineTaskAnnotationWriteOutcome,
    MachineTaskAnnotationWriteResult,
    RunDescriptor,
    Tag,
    TaskAnnotation,
    TaskAnnotationOrigin,
    TaskAnnotationProvenance,
    TaskIdentity,
    Verdict,
    decode_task_annotation_provenance,
    encode_task_annotation_provenance,
    normalize_annotation_tag_ids,
    normalize_tag_name,
    task_annotation_provenance_json,
    validate_annotation_note,
    validate_task_annotation,
    validate_task_identity,
    validate_task_tag_ids,
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
_PRE_IDENTITY_TASK_ANNOTATION_SCHEMA = (
    """
    CREATE TABLE registered_tasks (
        run_id VARCHAR NOT NULL,
        dataset_id VARCHAR NOT NULL,
        task_id VARCHAR NOT NULL,
        PRIMARY KEY (run_id, dataset_id, task_id)
    )
    """,
    """
    CREATE TABLE task_annotations (
        dataset_id VARCHAR NOT NULL CHECK (
            length(dataset_id) BETWEEN 1 AND 256
        ),
        task_id VARCHAR NOT NULL CHECK (length(task_id) BETWEEN 1 AND 256),
        origin VARCHAR NOT NULL CHECK (origin IN ('human', 'machine')),
        category VARCHAR CHECK (length(category) <= 256),
        note VARCHAR CHECK (length(note) <= 10000),
        provenance VARCHAR,
        created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
        PRIMARY KEY (dataset_id, task_id)
    )
    """,
    """
    CREATE TABLE task_annotation_tags (
        dataset_id VARCHAR NOT NULL,
        task_id VARCHAR NOT NULL,
        tag_id VARCHAR NOT NULL REFERENCES tags(tag_id),
        PRIMARY KEY (dataset_id, task_id, tag_id)
    )
    """,
)
_TASK_ANNOTATION_SCHEMA = (
    """
    CREATE TABLE registered_tasks (
        run_id VARCHAR NOT NULL,
        dataset_id VARCHAR NOT NULL,
        task_id VARCHAR NOT NULL,
        task_identity VARCHAR NOT NULL CHECK (
            regexp_full_match(task_identity, '[0-9a-f]{64}')
        ),
        PRIMARY KEY (run_id, dataset_id, task_id, task_identity),
        UNIQUE (run_id, dataset_id, task_id),
        UNIQUE (run_id, dataset_id, task_identity)
    )
    """,
    """
    CREATE TABLE task_annotations (
        dataset_id VARCHAR NOT NULL CHECK (
            length(dataset_id) BETWEEN 1 AND 256
        ),
        task_id VARCHAR NOT NULL CHECK (length(task_id) BETWEEN 1 AND 256),
        task_identity VARCHAR NOT NULL CHECK (
            regexp_full_match(task_identity, '[0-9a-f]{64}')
        ),
        origin VARCHAR NOT NULL CHECK (origin IN ('human', 'machine')),
        category VARCHAR CHECK (length(category) <= 256),
        note VARCHAR CHECK (length(note) <= 10000),
        provenance VARCHAR,
        created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
        PRIMARY KEY (dataset_id, task_id, task_identity)
    )
    """,
    """
    CREATE TABLE task_annotation_tags (
        dataset_id VARCHAR NOT NULL,
        task_id VARCHAR NOT NULL,
        task_identity VARCHAR NOT NULL,
        tag_id VARCHAR NOT NULL REFERENCES tags(tag_id),
        PRIMARY KEY (dataset_id, task_id, task_identity, tag_id)
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
_PRE_IDENTITY_TASK_ARCHIVE_SCHEMA = (
    """
    CREATE TABLE archived_registered_tasks (
        run_id VARCHAR NOT NULL,
        dataset_id VARCHAR NOT NULL,
        task_id VARCHAR NOT NULL,
        archive_reason VARCHAR NOT NULL,
        source_schema_version INTEGER NOT NULL,
        PRIMARY KEY (run_id, dataset_id, task_id)
    )
    """,
    """
    CREATE TABLE archived_task_annotations (
        dataset_id VARCHAR NOT NULL,
        task_id VARCHAR NOT NULL,
        origin VARCHAR NOT NULL,
        category VARCHAR,
        note VARCHAR,
        provenance VARCHAR,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        archive_reason VARCHAR NOT NULL,
        source_schema_version INTEGER NOT NULL,
        PRIMARY KEY (dataset_id, task_id)
    )
    """,
    """
    CREATE TABLE archived_task_annotation_tags (
        dataset_id VARCHAR NOT NULL,
        task_id VARCHAR NOT NULL,
        tag_id VARCHAR NOT NULL,
        tag_name VARCHAR NOT NULL,
        normalized_tag_name VARCHAR NOT NULL,
        tag_created_at TIMESTAMPTZ NOT NULL,
        archive_reason VARCHAR NOT NULL,
        source_schema_version INTEGER NOT NULL,
        PRIMARY KEY (dataset_id, task_id, tag_id)
    )
    """,
)
_ANNOTATION_ARCHIVE_SCHEMA = (*_LEGACY_SCHEMA, *_ARCHIVE_SCHEMA)
_TASK_ANNOTATION_DATABASE_SCHEMA = (
    *_LEGACY_SCHEMA,
    *_PRE_IDENTITY_TASK_ANNOTATION_SCHEMA,
)
_AUTHENTICATED_TASK_ANNOTATION_DATABASE_SCHEMA = (
    *_LEGACY_SCHEMA,
    *_TASK_ANNOTATION_SCHEMA,
)
_AUTHENTICATED_ANNOTATION_ARCHIVE_SCHEMA = (
    *_ANNOTATION_ARCHIVE_SCHEMA,
    *_TASK_ANNOTATION_SCHEMA,
)
_SCHEMA_V3 = (
    *_ANNOTATION_ARCHIVE_SCHEMA,
    *_PRE_IDENTITY_TASK_ANNOTATION_SCHEMA,
)
_SCHEMA = (
    *_ANNOTATION_ARCHIVE_SCHEMA,
    *_PRE_IDENTITY_TASK_ARCHIVE_SCHEMA,
    *_TASK_ANNOTATION_SCHEMA,
)
_VIEWER_SCHEMA: Final = """
    CREATE TABLE viewer_schema (
        schema_version INTEGER PRIMARY KEY
    )
    """
_LEGACY_SCHEMA_TABLES: Final = frozenset(
    {"runs", "annotations", "tags", "annotation_tags"}
)
_ARCHIVE_SCHEMA_TABLES: Final = frozenset(
    {
        "archived_annotations",
        "archived_annotation_tags",
        "archived_tags",
    }
)
_PRE_IDENTITY_TASK_ARCHIVE_SCHEMA_TABLES: Final = frozenset(
    {
        "archived_registered_tasks",
        "archived_task_annotations",
        "archived_task_annotation_tags",
    }
)
_TASK_ANNOTATION_SCHEMA_TABLES: Final = frozenset(
    {"registered_tasks", "task_annotations", "task_annotation_tags"}
)
_ANNOTATION_ARCHIVE_SCHEMA_TABLES: Final = (
    _LEGACY_SCHEMA_TABLES | _ARCHIVE_SCHEMA_TABLES
)
_TASK_ANNOTATION_DATABASE_SCHEMA_TABLES: Final = (
    _LEGACY_SCHEMA_TABLES | _TASK_ANNOTATION_SCHEMA_TABLES
)
_SCHEMA_V3_TABLES: Final = (
    _ANNOTATION_ARCHIVE_SCHEMA_TABLES | _TASK_ANNOTATION_SCHEMA_TABLES
)
_SCHEMA_TABLES: Final = (
    _SCHEMA_V3_TABLES | _PRE_IDENTITY_TASK_ARCHIVE_SCHEMA_TABLES
)
_LEGACY_SCHEMA_VERSION: Final = 1
_ANNOTATION_ARCHIVE_SCHEMA_VERSION: Final = 2
_PRE_IDENTITY_TASK_SCHEMA_VERSION: Final = 3
_SCHEMA_VERSION: Final = 4
_OWNERSHIP_GUARD = threading.Lock()
_INITIALIZATION_GUARD = threading.Lock()


@verify(UNIQUE)
class _ArchiveReason(StrEnum):
    """Persisted reason literals; never iterate to build stored payloads."""

    MISSING_AUTHENTICATED_TASK_IDENTITY = "missing_authenticated_task_identity"
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


@verify(UNIQUE)
class _TaskSchemaKind(StrEnum):
    NONE = "none"
    PRE_IDENTITY = "pre_identity"
    AUTHENTICATED = "authenticated"


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


_TASK_ANNOTATION_WRITE_ATTEMPTS = 3
type _TaskAnnotationRow = tuple[
    str,
    str,
    str,
    str,
    str | None,
    str | None,
    str | None,
    list[str],
    list[str],
]


@contextmanager
def _captured_registration_memberships(
    descriptors: tuple[RunDescriptor, ...],
) -> Iterator[Mapping[str, Path]]:
    capture_paths: dict[str, Path] = {}
    for descriptor in descriptors:
        if descriptor.evaluation_coordinates is None:
            continue
        membership_path = descriptor.candidate_membership_path
        if membership_path is None:
            raise InvalidQueryError(
                f"run {descriptor.run_id!r} has no candidate membership artifact"
            )
        capture_paths[descriptor.run_id] = membership_path

    with ExitStack() as stack:
        try:
            captured = stack.enter_context(stable_files(capture_paths))
        except (OSError, ValueError) as exc:
            raise InvalidQueryError(
                "cannot capture candidate membership for registration"
            ) from exc
        membership_paths: dict[str, Path] = {}
        for descriptor in descriptors:
            if descriptor.evaluation_coordinates is None:
                continue
            expected_sha256 = descriptor.artifact_sha256.get(
                "candidate_membership"
            )
            if (
                expected_sha256 is None
                or captured[descriptor.run_id].sha256 != expected_sha256
            ):
                raise InvalidQueryError(
                    f"run {descriptor.run_id!r} candidate membership does not "
                    "match authenticated descriptor"
                )
            membership_paths[descriptor.run_id] = captured[
                descriptor.run_id
            ].path
        yield membership_paths


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

        with _captured_registration_memberships(values) as memberships:

            def replace() -> None:
                for descriptor in values:
                    self.connection.execute(
                        "DELETE FROM registered_tasks WHERE run_id = ?",
                        [descriptor.run_id],
                    )
                    self._register_run(
                        descriptor, memberships.get(descriptor.run_id)
                    )

            self._transaction(replace)

    def task_is_registered(self, identity: TaskIdentity) -> bool:
        validated = validate_task_identity(
            identity.dataset_id,
            identity.task_id,
            identity.task_identity,
        )
        row = self.connection.execute(
            """
            SELECT 1
            FROM registered_tasks
            WHERE dataset_id = ? AND task_id = ? AND task_identity = ?
            LIMIT 1
            """,
            [
                validated.dataset_id,
                validated.task_id,
                validated.task_identity,
            ],
        ).fetchone()
        return row is not None

    def task_identity_for_run(
        self,
        run_id: str,
        dataset_id: str,
        task_id: str,
    ) -> str | None:
        """Return the candidate-evaluation identity registered for one run."""
        row = self.connection.execute(
            """
            SELECT task_identity
            FROM registered_tasks
            WHERE run_id = ? AND dataset_id = ? AND task_id = ?
            """,
            [run_id, dataset_id, task_id],
        ).fetchone()
        return row[0] if row is not None else None

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

    def get_task_annotation(
        self,
        dataset_id: str,
        task_id: str,
        task_identity: str,
    ) -> TaskAnnotation | None:
        identity = validate_task_identity(dataset_id, task_id, task_identity)
        return self._get_task_annotation(identity)

    def put_task_annotation(
        self,
        dataset_id: str,
        task_id: str,
        task_identity: str,
        *,
        category: str | None = None,
        note: str | None = None,
        tag_ids: Iterable[str] = (),
    ) -> TaskAnnotation:
        """Write a browser annotation, always as human without provenance."""
        identity = validate_task_identity(dataset_id, task_id, task_identity)
        selected_tag_ids = validate_task_tag_ids(tag_ids)
        candidate = validate_task_annotation(
            identity=identity,
            origin=TaskAnnotationOrigin.HUMAN,
            category=category,
            note=note,
            tags=(),
            provenance=None,
        )
        written: TaskAnnotation | None = None

        def upsert() -> None:
            nonlocal written
            self._validate_known_tags(selected_tag_ids)
            self._upsert_task_annotation_row(candidate)
            self._replace_task_annotation_tags(identity, selected_tag_ids)
            written = self._get_task_annotation(identity)

        self._retry_task_annotation_write(upsert)
        assert written is not None
        return written

    def put_machine_task_annotation(
        self,
        dataset_id: str,
        task_id: str,
        task_identity: str,
        *,
        category: str | None,
        note: str | None = None,
        tag_ids: Iterable[str] = (),
        provenance: TaskAnnotationProvenance,
    ) -> MachineTaskAnnotationWriteResult:
        """Atomically write machine state unless a human row is protected."""
        identity = validate_task_identity(dataset_id, task_id, task_identity)
        selected_tag_ids = validate_task_tag_ids(tag_ids)
        candidate = validate_task_annotation(
            identity=identity,
            origin=TaskAnnotationOrigin.MACHINE,
            category=category,
            note=note,
            tags=(),
            provenance=provenance,
        )
        result: MachineTaskAnnotationWriteResult | None = None

        def protect_and_upsert() -> None:
            nonlocal result
            self._validate_known_tags(selected_tag_ids)
            row = self._upsert_task_annotation_row(candidate)
            outcome = (
                MachineTaskAnnotationWriteOutcome.WRITTEN
                if row
                else MachineTaskAnnotationWriteOutcome.PROTECTED
            )
            if outcome is MachineTaskAnnotationWriteOutcome.WRITTEN:
                self._replace_task_annotation_tags(identity, selected_tag_ids)
            stored = self._get_task_annotation(identity)
            assert stored is not None
            result = MachineTaskAnnotationWriteResult(
                outcome=outcome,
                annotation=stored,
            )

        self._retry_task_annotation_write(protect_and_upsert)
        assert result is not None
        return result

    def delete_task_annotation(
        self,
        dataset_id: str,
        task_id: str,
        task_identity: str,
    ) -> bool:
        identity = validate_task_identity(dataset_id, task_id, task_identity)
        deleted = False

        def remove() -> None:
            nonlocal deleted
            self.connection.execute(
                """
                DELETE FROM task_annotation_tags
                WHERE dataset_id = ? AND task_id = ? AND task_identity = ?
                """,
                [
                    identity.dataset_id,
                    identity.task_id,
                    identity.task_identity,
                ],
            )
            row = self.connection.execute(
                """
                DELETE FROM task_annotations
                WHERE dataset_id = ? AND task_id = ? AND task_identity = ?
                RETURNING task_id
                """,
                [
                    identity.dataset_id,
                    identity.task_id,
                    identity.task_identity,
                ],
            ).fetchone()
            deleted = row is not None

        self._transaction(remove)
        return deleted

    def export_task_annotations(self) -> list[dict[str, object]]:
        """Return deterministic nested task annotations without timestamps."""
        exported: list[dict[str, object]] = []
        for annotation in self._task_annotations():
            exported.append(
                {
                    "identity": {
                        "dataset_id": annotation.identity.dataset_id,
                        "task_id": annotation.identity.task_id,
                        "task_identity": annotation.identity.task_identity,
                    },
                    "origin": annotation.origin.value,
                    "category": annotation.category,
                    "note": annotation.note,
                    "tags": [tag.name for tag in annotation.tags],
                    "provenance": (
                        task_annotation_provenance_json(annotation.provenance)
                        if annotation.provenance is not None
                        else None
                    ),
                }
            )
        return exported

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

    def _get_task_annotation(
        self, identity: TaskIdentity
    ) -> TaskAnnotation | None:
        annotations = self._task_annotations(identity)
        return annotations[0] if annotations else None

    def _task_annotations(
        self, identity: TaskIdentity | None = None
    ) -> tuple[TaskAnnotation, ...]:
        where = ""
        parameters: list[str] = []
        if identity is not None:
            where = (
                "WHERE a.dataset_id = ? AND a.task_id = ? "
                "AND a.task_identity = ?"
            )
            parameters = [
                identity.dataset_id,
                identity.task_id,
                identity.task_identity,
            ]
        rows = cast(
            list[_TaskAnnotationRow],
            self.connection.execute(
                f"""
                SELECT
                    a.dataset_id,
                    a.task_id,
                    a.task_identity,
                    a.origin,
                    a.category,
                    a.note,
                    a.provenance,
                    coalesce(
                        list(t.tag_id ORDER BY t.normalized_name, t.tag_id)
                            FILTER (WHERE t.tag_id IS NOT NULL),
                        []
                    ) AS tag_ids,
                    coalesce(
                        list(t.name ORDER BY t.normalized_name, t.tag_id)
                            FILTER (WHERE t.tag_id IS NOT NULL),
                        []
                    ) AS tag_names
                FROM task_annotations AS a
                LEFT JOIN task_annotation_tags AS atag
                  USING (dataset_id, task_id, task_identity)
                LEFT JOIN tags AS t USING (tag_id)
                {where}
                GROUP BY
                    a.dataset_id,
                    a.task_id,
                    a.task_identity,
                    a.origin,
                    a.category,
                    a.note,
                    a.provenance
                ORDER BY a.dataset_id, a.task_id, a.task_identity
                """,
                parameters,
            ).fetchall(),
        )
        return tuple(_decode_task_annotation_row(row) for row in rows)

    def _upsert_task_annotation_row(self, annotation: TaskAnnotation) -> bool:
        row = self.connection.execute(
            """
            INSERT INTO task_annotations(
                dataset_id, task_id, task_identity, origin, category, note,
                provenance
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id, task_id, task_identity) DO UPDATE SET
                origin = excluded.origin,
                category = excluded.category,
                note = excluded.note,
                provenance = excluded.provenance,
                updated_at = now()
            WHERE
                excluded.origin = 'human'
                OR task_annotations.origin = 'machine'
            RETURNING task_id
            """,
            [
                annotation.identity.dataset_id,
                annotation.identity.task_id,
                annotation.identity.task_identity,
                annotation.origin.value,
                annotation.category,
                annotation.note,
                (
                    encode_task_annotation_provenance(annotation.provenance)
                    if annotation.provenance is not None
                    else None
                ),
            ],
        ).fetchone()
        return row is not None

    def _replace_task_annotation_tags(
        self, identity: TaskIdentity, tag_ids: tuple[str, ...]
    ) -> None:
        self.connection.execute(
            """
            DELETE FROM task_annotation_tags
            WHERE dataset_id = ? AND task_id = ? AND task_identity = ?
            """,
            [
                identity.dataset_id,
                identity.task_id,
                identity.task_identity,
            ],
        )
        for tag_id in tag_ids:
            self.connection.execute(
                """
                INSERT INTO task_annotation_tags(
                    dataset_id, task_id, task_identity, tag_id
                )
                VALUES (?, ?, ?, ?)
                """,
                [
                    identity.dataset_id,
                    identity.task_id,
                    identity.task_identity,
                    tag_id,
                ],
            )

    def _validate_known_tags(self, tag_ids: tuple[str, ...]) -> None:
        if not tag_ids:
            return
        placeholders = ", ".join("?" for _ in tag_ids)
        found = {
            row[0]
            for row in self.connection.execute(
                f"SELECT tag_id FROM tags WHERE tag_id IN ({placeholders})",
                list(tag_ids),
            ).fetchall()
        }
        missing = sorted(set(tag_ids).difference(found))
        if missing:
            raise InvalidTaskAnnotationError(
                "unknown tag ID(s): " + ", ".join(missing)
            )

    def _register_run(
        self,
        descriptor: RunDescriptor,
        membership_path: Path | None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO runs(
                run_id, label, descriptor_json, manifest_sha256, corpus_sha256,
                definition_id
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
        if descriptor.evaluation_coordinates is None:
            return
        if membership_path is None:
            raise AssertionError("candidate membership was not captured")
        membership_schema = pq.ParquetFile(membership_path).schema_arrow
        for field_name in ("task_id", "task_identity"):
            field_type = membership_schema.field(field_name).type
            if not (
                pa.types.is_string(field_type)
                or pa.types.is_large_string(field_type)
            ):
                raise InvalidTaskAnnotationError(
                    f"{field_name} must be a string"
                )
        task_batches = self.connection.execute(
            """
            SELECT DISTINCT task_id, task_identity
            FROM read_parquet(?)
            """,
            [str(membership_path)],
        ).to_arrow_reader(4_096)
        for batch in task_batches:
            for task_id, task_identity in zip(
                batch.column(0), batch.column(1), strict=True
            ):
                validate_task_identity(
                    descriptor.dataset_id,
                    task_id.as_py(),
                    task_identity.as_py(),
                )
        ambiguous = self.connection.execute(
            """
            SELECT task_id
            FROM read_parquet(?)
            GROUP BY task_id
            HAVING count(DISTINCT task_identity) != 1
            LIMIT 1
            """,
            [str(membership_path)],
        ).fetchone()
        if ambiguous is not None:
            raise InvalidTaskAnnotationError(
                "candidate evaluation maps one task_id to multiple "
                "task_identity values"
            )
        reused_identity = self.connection.execute(
            """
            SELECT task_identity
            FROM read_parquet(?)
            GROUP BY task_identity
            HAVING count(DISTINCT task_id) != 1
            LIMIT 1
            """,
            [str(membership_path)],
        ).fetchone()
        if reused_identity is not None:
            raise InvalidTaskAnnotationError(
                "candidate evaluation maps one task_identity to multiple "
                "task_id values"
            )
        self.connection.execute(
            """
            INSERT INTO registered_tasks(
                run_id, dataset_id, task_id, task_identity
            )
            SELECT DISTINCT ?, ?, task_id, task_identity
            FROM read_parquet(?)
            """,
            [
                descriptor.run_id,
                descriptor.dataset_id,
                str(membership_path),
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
            self.connection.execute("COMMIT")
        except BaseException:
            try:
                self.connection.execute("ROLLBACK")
            except duckdb.TransactionException as rollback_error:
                if not _is_inactive_transaction_error(rollback_error):
                    raise
            raise

    def _retry_task_annotation_write(
        self, operation: Callable[[], object]
    ) -> None:
        for attempt in range(_TASK_ANNOTATION_WRITE_ATTEMPTS):
            try:
                self._transaction(operation)
            except duckdb.Error as error:
                if (
                    not _is_retryable_task_annotation_write_conflict(error)
                    or attempt + 1 == _TASK_ANNOTATION_WRITE_ATTEMPTS
                ):
                    raise
                continue
            return
        raise AssertionError("task annotation retry loop exhausted")


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
        _admit_versioned_schema(connection, tables)
        return

    if not catalog_objects:
        for statement in _SCHEMA:
            connection.execute(statement)
        _create_viewer_schema(connection, _SCHEMA_VERSION)
        return

    if tables == _LEGACY_SCHEMA_TABLES:
        source_schema = _LEGACY_SCHEMA
        source_schema_version = _LEGACY_SCHEMA_VERSION
        has_archive_tables = False
        task_schema_kind = _TaskSchemaKind.NONE
    elif tables == _TASK_ANNOTATION_DATABASE_SCHEMA_TABLES:
        if _schema_matches(
            connection,
            tables,
            tables,
            _TASK_ANNOTATION_DATABASE_SCHEMA,
        ):
            source_schema = _TASK_ANNOTATION_DATABASE_SCHEMA
            task_schema_kind = _TaskSchemaKind.PRE_IDENTITY
        elif _schema_matches(
            connection,
            tables,
            tables,
            _AUTHENTICATED_TASK_ANNOTATION_DATABASE_SCHEMA,
        ):
            source_schema = _AUTHENTICATED_TASK_ANNOTATION_DATABASE_SCHEMA
            task_schema_kind = _TaskSchemaKind.AUTHENTICATED
        else:
            _validate_schema_tables(
                connection,
                tables,
                tables,
                _TASK_ANNOTATION_DATABASE_SCHEMA,
            )
            raise AssertionError("unreachable malformed task schema")
        source_schema_version = _LEGACY_SCHEMA_VERSION
        has_archive_tables = False
    elif tables == _ANNOTATION_ARCHIVE_SCHEMA_TABLES:
        source_schema = _ANNOTATION_ARCHIVE_SCHEMA
        source_schema_version = _ANNOTATION_ARCHIVE_SCHEMA_VERSION
        has_archive_tables = True
        task_schema_kind = _TaskSchemaKind.NONE
    elif tables == _SCHEMA_V3_TABLES:
        if _schema_matches(connection, tables, tables, _SCHEMA_V3):
            source_schema = _SCHEMA_V3
            task_schema_kind = _TaskSchemaKind.PRE_IDENTITY
        elif _schema_matches(
            connection,
            tables,
            tables,
            _AUTHENTICATED_ANNOTATION_ARCHIVE_SCHEMA,
        ):
            source_schema = _AUTHENTICATED_ANNOTATION_ARCHIVE_SCHEMA
            task_schema_kind = _TaskSchemaKind.AUTHENTICATED
        else:
            _validate_schema_tables(
                connection,
                tables,
                tables,
                _SCHEMA_V3,
            )
            raise AssertionError("unreachable malformed composed schema")
        source_schema_version = _PRE_IDENTITY_TASK_SCHEMA_VERSION
        has_archive_tables = True
    else:
        _raise_unsupported_table_set(
            "unversioned schema", tables, _LEGACY_SCHEMA_TABLES
        )
        return

    # Prove the entire recognized unversioned shape before adding metadata.
    _validate_schema_tables(
        connection,
        tables,
        tables,
        source_schema,
    )
    _migrate_to_current_schema(
        connection,
        source_schema_version=source_schema_version,
        has_archive_tables=has_archive_tables,
        task_schema_kind=task_schema_kind,
        create_metadata=True,
    )


def _admit_versioned_schema(
    connection: duckdb.DuckDBPyConnection,
    tables: frozenset[str],
) -> None:
    _validate_table_schema(connection, "viewer_schema", _SCHEMA)
    versions = connection.execute(
        "SELECT schema_version FROM viewer_schema"
    ).fetchall()
    current_tables = _SCHEMA_TABLES | {"viewer_schema"}
    if versions == [(_SCHEMA_VERSION,)]:
        _validate_schema_tables(connection, tables, current_tables, _SCHEMA)
        _archive_out_of_contract_rows(connection, _SCHEMA_VERSION)
        return

    annotation_archive_tables = _ANNOTATION_ARCHIVE_SCHEMA_TABLES | {
        "viewer_schema"
    }
    legacy_tables = _LEGACY_SCHEMA_TABLES | {"viewer_schema"}
    task_tables = _TASK_ANNOTATION_DATABASE_SCHEMA_TABLES | {"viewer_schema"}
    composed_tables = _SCHEMA_V3_TABLES | {"viewer_schema"}
    if versions == [(_ANNOTATION_ARCHIVE_SCHEMA_VERSION,)] and (
        tables == annotation_archive_tables
    ):
        source_schema = _ANNOTATION_ARCHIVE_SCHEMA
        has_archive_tables = True
        task_schema_kind = _TaskSchemaKind.NONE
    elif (
        versions
        in (
            [(_LEGACY_SCHEMA_VERSION,)],
            [(_ANNOTATION_ARCHIVE_SCHEMA_VERSION,)],
        )
        and tables == task_tables
    ):
        if _schema_matches(
            connection,
            tables,
            task_tables,
            _TASK_ANNOTATION_DATABASE_SCHEMA,
        ):
            source_schema = _TASK_ANNOTATION_DATABASE_SCHEMA
            task_schema_kind = _TaskSchemaKind.PRE_IDENTITY
        elif _schema_matches(
            connection,
            tables,
            task_tables,
            _AUTHENTICATED_TASK_ANNOTATION_DATABASE_SCHEMA,
        ):
            source_schema = _AUTHENTICATED_TASK_ANNOTATION_DATABASE_SCHEMA
            task_schema_kind = _TaskSchemaKind.AUTHENTICATED
        else:
            raise DatabaseSchemaError(
                "viewer database historical task schema is malformed"
            )
        has_archive_tables = False
    elif (
        versions
        in (
            [(_LEGACY_SCHEMA_VERSION,)],
            [(_ANNOTATION_ARCHIVE_SCHEMA_VERSION,)],
            [(_PRE_IDENTITY_TASK_SCHEMA_VERSION,)],
        )
        and tables == composed_tables
    ):
        if _schema_matches(connection, tables, composed_tables, _SCHEMA_V3):
            source_schema = _SCHEMA_V3
            task_schema_kind = _TaskSchemaKind.PRE_IDENTITY
        elif _schema_matches(
            connection,
            tables,
            composed_tables,
            _AUTHENTICATED_ANNOTATION_ARCHIVE_SCHEMA,
        ):
            source_schema = _AUTHENTICATED_ANNOTATION_ARCHIVE_SCHEMA
            task_schema_kind = _TaskSchemaKind.AUTHENTICATED
        else:
            raise DatabaseSchemaError(
                "viewer database historical composed schema is malformed"
            )
        has_archive_tables = True
    elif versions == [(_LEGACY_SCHEMA_VERSION,)] and tables == legacy_tables:
        source_schema = _LEGACY_SCHEMA
        has_archive_tables = False
        task_schema_kind = _TaskSchemaKind.NONE
    elif tables == _LEGACY_SCHEMA_TABLES:
        raise AssertionError("versioned schema omitted viewer_schema")
    else:
        raise DatabaseSchemaError(
            "viewer database schema version/table signature is unsupported: "
            f"version rows {versions!r}, tables {sorted(tables)!r}"
        )

    _validate_schema_tables(
        connection,
        tables,
        frozenset(tables),
        source_schema,
    )
    _migrate_to_current_schema(
        connection,
        source_schema_version=versions[0][0],
        has_archive_tables=has_archive_tables,
        task_schema_kind=task_schema_kind,
        create_metadata=False,
    )


def _migrate_to_current_schema(
    connection: duckdb.DuckDBPyConnection,
    *,
    source_schema_version: int,
    has_archive_tables: bool,
    task_schema_kind: _TaskSchemaKind,
    create_metadata: bool,
) -> None:
    if task_schema_kind is _TaskSchemaKind.AUTHENTICATED:
        _reject_out_of_contract_task_tag_links(connection)
    if not has_archive_tables:
        for statement in _ARCHIVE_SCHEMA:
            connection.execute(statement)
    for statement in _PRE_IDENTITY_TASK_ARCHIVE_SCHEMA:
        connection.execute(statement)
    if task_schema_kind is _TaskSchemaKind.PRE_IDENTITY:
        _archive_pre_identity_task_rows(connection, source_schema_version)
    if task_schema_kind is not _TaskSchemaKind.AUTHENTICATED:
        for statement in _TASK_ANNOTATION_SCHEMA:
            connection.execute(statement)
    _archive_out_of_contract_rows(connection, source_schema_version)
    if create_metadata:
        _create_viewer_schema(connection, _SCHEMA_VERSION)
        return
    connection.execute(
        "UPDATE viewer_schema SET schema_version = ?",
        [_SCHEMA_VERSION],
    )


def _archive_pre_identity_task_rows(
    connection: duckdb.DuckDBPyConnection,
    source_schema_version: int,
) -> None:
    reason = _ArchiveReason.MISSING_AUTHENTICATED_TASK_IDENTITY.value
    source_counts: dict[str, int] = {}
    for table in (
        "registered_tasks",
        "task_annotations",
        "task_annotation_tags",
    ):
        row = connection.execute(f"SELECT count(*) FROM {table}").fetchone()
        assert row is not None
        source_counts[table] = cast(int, row[0])
    connection.execute(
        """
        INSERT INTO archived_registered_tasks(
            run_id,
            dataset_id,
            task_id,
            archive_reason,
            source_schema_version
        )
        SELECT run_id, dataset_id, task_id, ?, ?
        FROM registered_tasks
        """,
        [reason, source_schema_version],
    )
    connection.execute(
        """
        INSERT INTO archived_task_annotations(
            dataset_id,
            task_id,
            origin,
            category,
            note,
            provenance,
            created_at,
            updated_at,
            archive_reason,
            source_schema_version
        )
        SELECT
            dataset_id,
            task_id,
            origin,
            category,
            note,
            provenance,
            created_at,
            updated_at,
            ?,
            ?
        FROM task_annotations
        """,
        [reason, source_schema_version],
    )
    connection.execute(
        """
        INSERT INTO archived_task_annotation_tags(
            dataset_id,
            task_id,
            tag_id,
            tag_name,
            normalized_tag_name,
            tag_created_at,
            archive_reason,
            source_schema_version
        )
        SELECT
            link.dataset_id,
            link.task_id,
            link.tag_id,
            tag.name,
            tag.normalized_name,
            tag.created_at,
            ?,
            ?
        FROM task_annotation_tags AS link
        JOIN tags AS tag USING (tag_id)
        """,
        [reason, source_schema_version],
    )
    archive_tables = {
        "registered_tasks": "archived_registered_tasks",
        "task_annotations": "archived_task_annotations",
        "task_annotation_tags": "archived_task_annotation_tags",
    }
    for source_table, archive_table in archive_tables.items():
        row = connection.execute(
            f"SELECT count(*) FROM {archive_table}"
        ).fetchone()
        assert row is not None
        archived_count = cast(int, row[0])
        if archived_count != source_counts[source_table]:
            raise DatabaseSchemaError(
                "pre-identity task migration did not preserve every row from "
                f"{source_table}"
            )
    connection.execute("DROP TABLE task_annotation_tags")
    connection.execute("DROP TABLE task_annotations")
    connection.execute("DROP TABLE registered_tasks")


def _create_viewer_schema(
    connection: duckdb.DuckDBPyConnection, schema_version: int
) -> None:
    connection.execute(_VIEWER_SCHEMA)
    connection.execute(
        "INSERT INTO viewer_schema(schema_version) VALUES (?)",
        [schema_version],
    )


def _raise_unsupported_table_set(
    label: str,
    tables: frozenset[str],
    expected_tables: frozenset[str],
) -> None:
    missing = sorted(expected_tables.difference(tables))
    unexpected = sorted(tables.difference(expected_tables))
    details = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    raise DatabaseSchemaError(
        f"viewer database {label} is unsupported: " + "; ".join(details)
    )


def _reject_out_of_contract_task_tag_links(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    if "task_annotation_tags" not in _main_tables(connection):
        return
    invalid_tag_ids = [
        row[0]
        for row in connection.execute(
            "SELECT tag_id, name, normalized_name FROM tags ORDER BY tag_id"
        ).fetchall()
        if _tag_archive_reasons(row[1], row[2])
    ]
    if not invalid_tag_ids:
        return
    placeholders = ", ".join("?" for _tag_id in invalid_tag_ids)
    links = connection.execute(
        f"""
        SELECT dataset_id, task_id, tag_id
        FROM task_annotation_tags
        WHERE tag_id IN ({placeholders})
        ORDER BY dataset_id, task_id, tag_id
        """,
        invalid_tag_ids,
    ).fetchall()
    if links:
        raise DatabaseSchemaError(
            "task annotations reference out-of-contract tags and cannot be "
            "migrated without losing annotation closure: "
            f"{links!r}; export and repair those task annotations first"
        )


def _archive_out_of_contract_rows(
    connection: duckdb.DuckDBPyConnection, source_schema_version: int
) -> None:
    _reject_out_of_contract_task_tag_links(connection)
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
    _validate_schema_tables(connection, tables, expected_tables, _SCHEMA)
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
    schema: tuple[str, ...] = _SCHEMA,
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
        _validate_table_schema(connection, table, schema)


def _schema_matches(
    connection: duckdb.DuckDBPyConnection,
    tables: frozenset[str],
    expected_tables: frozenset[str],
    schema: tuple[str, ...],
) -> bool:
    return tables == expected_tables and all(
        _table_schema_signature(connection, table)
        == _expected_table_schema_signature(table, schema)
        for table in sorted(expected_tables)
    )


def _validate_table_schema(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    schema: tuple[str, ...] = _SCHEMA,
) -> None:
    signature = _table_schema_signature(connection, table)
    expected = _expected_table_schema_signature(table, schema)
    if signature != expected:
        raise DatabaseSchemaError(
            f"viewer database schema for {table} is malformed: "
            f"expected {expected!r}, found {signature!r}"
        )


@cache
def _expected_table_schema_signature(
    table: str, schema: tuple[str, ...] = _SCHEMA
) -> tuple[tuple[tuple[object, ...], ...], tuple[tuple[object, ...], ...]]:
    with duckdb.connect(":memory:") as expected:
        for statement in schema:
            expected.execute(statement)
        expected.execute(_VIEWER_SCHEMA)
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


def _is_retryable_task_annotation_write_conflict(
    error: duckdb.Error,
) -> bool:
    if isinstance(error, duckdb.TransactionException):
        return True
    message = str(error)
    return (
        isinstance(error, duckdb.ConstraintException)
        and 'Duplicate key "dataset_id:' in message
        and "task_id:" in message
        and "violates primary key constraint" in message
    )


def _is_inactive_transaction_error(error: duckdb.TransactionException) -> bool:
    return "no transaction is active" in str(error)


def _decode_task_annotation_row(row: _TaskAnnotationRow) -> TaskAnnotation:
    return validate_task_annotation(
        identity=validate_task_identity(row[0], row[1], row[2]),
        origin=row[3],
        category=row[4],
        note=row[5],
        tags=tuple(
            Tag(tag_id=tag_id, name=name)
            for tag_id, name in zip(row[7], row[8], strict=True)
        ),
        provenance=(
            decode_task_annotation_provenance(row[6])
            if row[6] is not None
            else None
        ),
    )


__all__ = (
    "DatabaseOwnershipError",
    "DatabaseSchemaError",
    "ViewerDatabase",
    "database_owner_lock_path",
    "database_ownership",
)
