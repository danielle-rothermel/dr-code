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
from typing import BinaryIO, Final, cast

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from dr_code.viewer.domain import (
    Annotation,
    InvalidQueryError,
    InvalidTaskAnnotationError,
    MachineTaskAnnotationBatchResult,
    MachineTaskAnnotationWriteOutcome,
    MachineTaskAnnotationWriteResult,
    RunDescriptor,
    Tag,
    TaskAnnotation,
    TaskAnnotationOrigin,
    TaskAnnotationPublicationIntent,
    TaskAnnotationProvenance,
    TaskIdentity,
    Verdict,
    decode_task_annotation_provenance,
    encode_task_annotation_provenance,
    task_annotation_provenance_json,
    validate_task_annotation,
    validate_task_identity,
    validate_task_tag_ids,
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
    CREATE TABLE IF NOT EXISTS registered_tasks (
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
    """
    CREATE TABLE IF NOT EXISTS task_annotations (
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
    CREATE TABLE IF NOT EXISTS task_annotation_tags (
        dataset_id VARCHAR NOT NULL,
        task_id VARCHAR NOT NULL,
        task_identity VARCHAR NOT NULL,
        tag_id VARCHAR NOT NULL REFERENCES tags(tag_id),
        PRIMARY KEY (dataset_id, task_id, task_identity, tag_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS task_annotation_publication_intents (
        producer VARCHAR NOT NULL,
        experiment_identity VARCHAR NOT NULL,
        output_path VARCHAR PRIMARY KEY,
        staged_path VARCHAR NOT NULL UNIQUE,
        prior_sha256 VARCHAR,
        intended_sha256 VARCHAR NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
    )
    """,
)
_SCHEMA_TABLES: Final = frozenset(
    {
        "runs",
        "registered_tasks",
        "annotations",
        "tags",
        "annotation_tags",
        "task_annotations",
        "task_annotation_tags",
        "task_annotation_publication_intents",
    }
)
_SCHEMA_VERSION: Final = 1
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


class DatabaseSchemaError(RuntimeError):
    """The database cannot be opened without ambiguous persisted state."""


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


_TASK_ANNOTATION_WRITE_ATTEMPTS = 3
_PUBLICATION_BATCH_SIZE = 128
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
type _PublicationIntentRow = tuple[
    str,
    str,
    str,
    str,
    str | None,
    str,
]


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

        def replace() -> None:
            for descriptor in values:
                self._connection.execute(
                    "DELETE FROM registered_tasks WHERE run_id = ?",
                    [descriptor.run_id],
                )
                self._register_run(descriptor)

        self._transaction(replace)

    def task_is_registered(self, identity: TaskIdentity) -> bool:
        validated = validate_task_identity(
            identity.dataset_id,
            identity.task_id,
            identity.task_identity,
        )
        row = self._connection.execute(
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
        row = self._connection.execute(
            """
            SELECT task_identity
            FROM registered_tasks
            WHERE run_id = ? AND dataset_id = ? AND task_id = ?
            """,
            [run_id, dataset_id, task_id],
        ).fetchone()
        return row[0] if row is not None else None

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
            stored = self._get_task_annotation(
                identity,
                suppress_pending_publications=False,
            )
            assert stored is not None
            result = MachineTaskAnnotationWriteResult(
                outcome=outcome,
                annotation=stored,
            )

        self._retry_task_annotation_write(protect_and_upsert)
        assert result is not None
        return result

    def begin_task_annotation_publication(
        self,
        intent: TaskAnnotationPublicationIntent,
    ) -> None:
        """Durably hide one experiment's machine rows before file publish."""
        validated = _validate_publication_intent(intent)

        def insert() -> None:
            self._connection.execute(
                """
                INSERT INTO task_annotation_publication_intents(
                    producer,
                    experiment_identity,
                    output_path,
                    staged_path,
                    prior_sha256,
                    intended_sha256
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    validated.producer,
                    validated.experiment_identity,
                    validated.output_path,
                    validated.staged_path,
                    validated.prior_sha256,
                    validated.intended_sha256,
                ],
            )

        self._retry_task_annotation_write(insert)

    def get_task_annotation_publication_intent(
        self,
        output_path: str,
    ) -> TaskAnnotationPublicationIntent | None:
        validated_path = _canonical_absolute_path(
            output_path,
            "publication output_path",
        )
        row = cast(
            _PublicationIntentRow | None,
            self._connection.execute(
                """
                SELECT
                    producer,
                    experiment_identity,
                    output_path,
                    staged_path,
                    prior_sha256,
                    intended_sha256
                FROM task_annotation_publication_intents
                WHERE output_path = ?
                """,
                [validated_path],
            ).fetchone(),
        )
        return _publication_intent_from_row(row) if row is not None else None

    def abort_task_annotation_publication(
        self,
        intent: TaskAnnotationPublicationIntent,
    ) -> None:
        """Delete exactly one unambiguous pre-replace publication intent."""
        validated = _validate_publication_intent(intent)

        def delete() -> None:
            self._require_publication_intent(validated)
            self._delete_publication_intent(validated)

        self._retry_task_annotation_write(delete)

    def finalize_task_annotation_publication(
        self,
        scope: Iterable[TaskIdentity],
        annotations: Iterable[TaskAnnotation],
        *,
        intent: TaskAnnotationPublicationIntent,
    ) -> MachineTaskAnnotationBatchResult:
        """Atomically replace exact rollups and consume their file intent."""
        validated_intent = _validate_publication_intent(intent)
        validated_producer = validated_intent.producer
        experiment_identity = validated_intent.experiment_identity
        stage = f"_task_annotation_publication_{uuid.uuid4().hex}"
        self._connection.execute(
            f"""
            CREATE TEMP TABLE {stage} (
                dataset_id VARCHAR NOT NULL,
                task_id VARCHAR NOT NULL,
                task_identity VARCHAR NOT NULL,
                has_annotation BOOLEAN NOT NULL DEFAULT false,
                category VARCHAR,
                note VARCHAR,
                provenance VARCHAR,
                PRIMARY KEY (dataset_id, task_id, task_identity)
            )
            """
        )

        result: MachineTaskAnnotationBatchResult | None = None

        def replace() -> None:
            nonlocal result
            self._require_publication_intent(validated_intent)
            written = 0
            protected = 0
            removed = 0
            for batch in self._publication_stage_batches(stage):
                for (
                    dataset_id,
                    task_id,
                    task_identity,
                    has_annotation,
                    category,
                    note,
                    provenance_json,
                ) in batch:
                    identity = TaskIdentity(
                        dataset_id,
                        task_id,
                        task_identity,
                    )
                    if has_annotation:
                        assert provenance_json is not None
                        candidate = TaskAnnotation(
                            identity=identity,
                            origin=TaskAnnotationOrigin.MACHINE,
                            category=category,
                            note=note,
                            tags=(),
                            provenance=decode_task_annotation_provenance(
                                provenance_json
                            ),
                        )
                        row = self._upsert_task_annotation_row(candidate)
                        outcome = (
                            MachineTaskAnnotationWriteOutcome.WRITTEN
                            if row
                            else MachineTaskAnnotationWriteOutcome.PROTECTED
                        )
                        if (
                            outcome
                            is MachineTaskAnnotationWriteOutcome.PROTECTED
                        ):
                            protected += 1
                        else:
                            self._replace_task_annotation_tags(identity, ())
                            written += 1
                        continue

                    row = self._connection.execute(
                        """
                        DELETE FROM task_annotations
                        WHERE dataset_id = ?
                          AND task_id = ?
                          AND task_identity = ?
                          AND origin = 'machine'
                          AND json_extract_string(
                              provenance, '$.extra.producer'
                          ) = ?
                          AND json_extract_string(
                              provenance, '$.extra.experiment_identity'
                          ) = ?
                        RETURNING task_id
                        """,
                        [
                            dataset_id,
                            task_id,
                            task_identity,
                            validated_producer,
                            experiment_identity,
                        ],
                    ).fetchone()
                    if row is not None:
                        self._connection.execute(
                            """
                            DELETE FROM task_annotation_tags
                            WHERE dataset_id = ?
                              AND task_id = ?
                              AND task_identity = ?
                            """,
                            [dataset_id, task_id, task_identity],
                        )
                        removed += 1
                        continue
                    existing = self._connection.execute(
                        """
                        SELECT origin FROM task_annotations
                        WHERE dataset_id = ?
                          AND task_id = ?
                          AND task_identity = ?
                        """,
                        [dataset_id, task_id, task_identity],
                    ).fetchone()
                    if existing is not None and existing[0] == "human":
                        protected += 1
            self._delete_publication_intent(validated_intent)
            result = MachineTaskAnnotationBatchResult(
                written=written,
                protected=protected,
                removed=removed,
            )

        try:
            self._stage_task_annotation_publication(
                stage,
                scope,
                annotations,
                producer=validated_producer,
                experiment_identity=experiment_identity,
            )
            self._retry_task_annotation_write(replace)
        finally:
            self._connection.execute(f"DROP TABLE IF EXISTS {stage}")
        assert result is not None
        return result

    def _stage_task_annotation_publication(
        self,
        stage: str,
        scope: Iterable[TaskIdentity],
        annotations: Iterable[TaskAnnotation],
        *,
        producer: str,
        experiment_identity: str,
    ) -> None:
        for identity_value in scope:
            identity = validate_task_identity(
                identity_value.dataset_id,
                identity_value.task_id,
                identity_value.task_identity,
            )
            try:
                self._connection.execute(
                    f"""
                    INSERT INTO {stage}(dataset_id, task_id, task_identity)
                    VALUES (?, ?, ?)
                    """,
                    [
                        identity.dataset_id,
                        identity.task_id,
                        identity.task_identity,
                    ],
                )
            except duckdb.ConstraintException as exc:
                raise InvalidTaskAnnotationError(
                    "machine replacement scope contains duplicate task "
                    "identities"
                ) from exc
        for annotation in annotations:
            if not isinstance(annotation, TaskAnnotation):
                raise InvalidTaskAnnotationError(
                    "machine replacements must contain TaskAnnotation values"
                )
            if annotation.origin is not TaskAnnotationOrigin.MACHINE:
                raise InvalidTaskAnnotationError(
                    "machine replacements must have machine origin"
                )
            if annotation.tags:
                raise InvalidTaskAnnotationError(
                    "machine replacements do not accept materialized tags"
                )
            identity = validate_task_identity(
                annotation.identity.dataset_id,
                annotation.identity.task_id,
                annotation.identity.task_identity,
            )
            assert annotation.provenance is not None
            if (
                annotation.provenance.extra.get("producer") != producer
                or annotation.provenance.extra.get("experiment_identity")
                != experiment_identity
            ):
                raise InvalidTaskAnnotationError(
                    "machine replacement provenance does not match its owner"
                )
            row = self._connection.execute(
                f"""
                UPDATE {stage}
                SET has_annotation = true,
                    category = ?,
                    note = ?,
                    provenance = ?
                WHERE dataset_id = ?
                  AND task_id = ?
                  AND task_identity = ?
                  AND NOT has_annotation
                RETURNING task_id
                """,
                [
                    annotation.category,
                    annotation.note,
                    encode_task_annotation_provenance(annotation.provenance),
                    identity.dataset_id,
                    identity.task_id,
                    identity.task_identity,
                ],
            ).fetchone()
            if row is not None:
                continue
            existing = self._connection.execute(
                f"""
                SELECT has_annotation FROM {stage}
                WHERE dataset_id = ?
                  AND task_id = ?
                  AND task_identity = ?
                """,
                [
                    identity.dataset_id,
                    identity.task_id,
                    identity.task_identity,
                ],
            ).fetchone()
            if existing is None:
                raise InvalidTaskAnnotationError(
                    "machine replacement is outside the selected task scope"
                )
            raise InvalidTaskAnnotationError(
                "machine replacements contain duplicate task identities"
            )

    def _publication_stage_batches(
        self,
        stage: str,
    ) -> Iterable[
        list[
            tuple[
                str,
                str,
                str,
                bool,
                str | None,
                str | None,
                str | None,
            ]
        ]
    ]:
        last: tuple[str, str, str] | None = None
        while True:
            if last is None:
                query = f"""
                    SELECT dataset_id, task_id, task_identity, has_annotation,
                           category, note, provenance
                    FROM {stage}
                    ORDER BY dataset_id, task_id, task_identity
                    LIMIT ?
                """
                parameters: list[object] = [_PUBLICATION_BATCH_SIZE]
            else:
                query = f"""
                    SELECT dataset_id, task_id, task_identity, has_annotation,
                           category, note, provenance
                    FROM {stage}
                    WHERE (dataset_id, task_id, task_identity) > (?, ?, ?)
                    ORDER BY dataset_id, task_id, task_identity
                    LIMIT ?
                """
                parameters = [*last, _PUBLICATION_BATCH_SIZE]
            rows = cast(
                list[
                    tuple[
                        str,
                        str,
                        str,
                        bool,
                        str | None,
                        str | None,
                        str | None,
                    ]
                ],
                self._connection.execute(query, parameters).fetchall(),
            )
            if not rows:
                return
            yield rows
            last = rows[-1][0], rows[-1][1], rows[-1][2]

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
            self._connection.execute(
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
            row = self._connection.execute(
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

    def _get_task_annotation(
        self,
        identity: TaskIdentity,
        *,
        suppress_pending_publications: bool = True,
    ) -> TaskAnnotation | None:
        annotations = self._task_annotations(
            identity,
            suppress_pending_publications=suppress_pending_publications,
        )
        return annotations[0] if annotations else None

    def _task_annotations(
        self,
        identity: TaskIdentity | None = None,
        *,
        suppress_pending_publications: bool = True,
    ) -> tuple[TaskAnnotation, ...]:
        conditions: list[str] = []
        parameters: list[str] = []
        if identity is not None:
            conditions.append(
                "a.dataset_id = ? AND a.task_id = ? AND a.task_identity = ?"
            )
            parameters = [
                identity.dataset_id,
                identity.task_id,
                identity.task_identity,
            ]
        if suppress_pending_publications:
            conditions.append(
                """
                (
                    a.origin = 'human'
                    OR NOT EXISTS (
                        SELECT 1
                        FROM task_annotation_publication_intents AS intent
                        WHERE intent.producer = try(
                            json_extract_string(
                                a.provenance, '$.extra.producer'
                            )
                        )
                          AND intent.experiment_identity = try(
                            json_extract_string(
                                a.provenance,
                                '$.extra.experiment_identity'
                            )
                        )
                    )
                )
                """
            )
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        rows = cast(
            list[_TaskAnnotationRow],
            self._connection.execute(
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

    def _require_publication_intent(
        self,
        intent: TaskAnnotationPublicationIntent,
    ) -> None:
        row = self._connection.execute(
            """
            SELECT 1
            FROM task_annotation_publication_intents
            WHERE producer = ?
              AND experiment_identity = ?
              AND output_path = ?
              AND staged_path = ?
              AND prior_sha256 IS NOT DISTINCT FROM ?
              AND intended_sha256 = ?
            """,
            _publication_intent_parameters(intent),
        ).fetchone()
        if row is None:
            raise InvalidTaskAnnotationError(
                "exact task annotation publication intent is not pending"
            )

    def _delete_publication_intent(
        self,
        intent: TaskAnnotationPublicationIntent,
    ) -> None:
        row = self._connection.execute(
            """
            DELETE FROM task_annotation_publication_intents
            WHERE producer = ?
              AND experiment_identity = ?
              AND output_path = ?
              AND staged_path = ?
              AND prior_sha256 IS NOT DISTINCT FROM ?
              AND intended_sha256 = ?
            RETURNING output_path
            """,
            _publication_intent_parameters(intent),
        ).fetchone()
        if row is None:
            raise InvalidTaskAnnotationError(
                "exact task annotation publication intent disappeared"
            )

    def _upsert_task_annotation_row(self, annotation: TaskAnnotation) -> bool:
        row = self._connection.execute(
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
        self._connection.execute(
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
            self._connection.execute(
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
            for row in self._connection.execute(
                f"SELECT tag_id FROM tags WHERE tag_id IN ({placeholders})",
                list(tag_ids),
            ).fetchall()
        }
        missing = sorted(set(tag_ids).difference(found))
        if missing:
            raise InvalidTaskAnnotationError(
                "unknown tag ID(s): " + ", ".join(missing)
            )

    def _register_run(self, descriptor: RunDescriptor) -> None:
        self._connection.execute(
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
        membership_path = descriptor.candidate_membership_path
        assert membership_path is not None
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
        task_batches = self._connection.execute(
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
        ambiguous = self._connection.execute(
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
        reused_identity = self._connection.execute(
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
        self._connection.execute(
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
        self._connection.execute("BEGIN TRANSACTION")
        try:
            operation()
            self._connection.execute("COMMIT")
        except BaseException:
            try:
                self._connection.execute("ROLLBACK")
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


def _nonblank_trimmed(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise InvalidTaskAnnotationError(
            f"{label} must be a nonblank trimmed string"
        )
    return value


def _validate_publication_intent(
    intent: TaskAnnotationPublicationIntent,
) -> TaskAnnotationPublicationIntent:
    if not isinstance(intent, TaskAnnotationPublicationIntent):
        raise InvalidTaskAnnotationError(
            "publication intent must be a TaskAnnotationPublicationIntent"
        )
    producer = _nonblank_trimmed(intent.producer, "machine producer")
    validate_sha256(intent.experiment_identity, "experiment_identity")
    output_path = _canonical_absolute_path(
        intent.output_path,
        "publication output_path",
    )
    staged_path = _canonical_absolute_path(
        intent.staged_path,
        "publication staged_path",
    )
    if output_path == staged_path:
        raise InvalidTaskAnnotationError(
            "publication output and staged paths must be different"
        )
    if Path(output_path).parent != Path(staged_path).parent:
        raise InvalidTaskAnnotationError(
            "publication output and staged paths must share a directory"
        )
    if intent.prior_sha256 is not None:
        validate_sha256(intent.prior_sha256, "prior_sha256")
    validate_sha256(intent.intended_sha256, "intended_sha256")
    return TaskAnnotationPublicationIntent(
        producer=producer,
        experiment_identity=intent.experiment_identity,
        output_path=output_path,
        staged_path=staged_path,
        prior_sha256=intent.prior_sha256,
        intended_sha256=intent.intended_sha256,
    )


def _canonical_absolute_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidTaskAnnotationError(
            f"{label} must be a canonical absolute path"
        )
    path = Path(value)
    if not path.is_absolute() or str(path.resolve()) != value:
        raise InvalidTaskAnnotationError(
            f"{label} must be a canonical absolute path"
        )
    return value


def _publication_intent_parameters(
    intent: TaskAnnotationPublicationIntent,
) -> list[str | None]:
    return [
        intent.producer,
        intent.experiment_identity,
        intent.output_path,
        intent.staged_path,
        intent.prior_sha256,
        intent.intended_sha256,
    ]


def _publication_intent_from_row(
    row: _PublicationIntentRow,
) -> TaskAnnotationPublicationIntent:
    return _validate_publication_intent(
        TaskAnnotationPublicationIntent(
            producer=row[0],
            experiment_identity=row[1],
            output_path=row[2],
            staged_path=row[3],
            prior_sha256=row[4],
            intended_sha256=row[5],
        )
    )


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
        _prepare_schema(connection)
        for statement in _SCHEMA:
            connection.execute(statement)
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")


def _prepare_schema(connection: duckdb.DuckDBPyConnection) -> None:
    tables = {
        row[0]
        for row in connection.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            """
        ).fetchall()
    }
    if "viewer_schema" in tables:
        versions = connection.execute(
            "SELECT schema_version FROM viewer_schema"
        ).fetchall()
        if versions != [(_SCHEMA_VERSION,)]:
            raise DatabaseSchemaError(
                "viewer database schema version is unsupported: "
                f"expected {_SCHEMA_VERSION}, found {versions!r}"
            )
        missing = sorted(_SCHEMA_TABLES.difference(tables))
        if missing:
            raise DatabaseSchemaError(
                "viewer database schema is incomplete: " + ", ".join(missing)
            )
        for table in (
            "registered_tasks",
            "task_annotations",
            "task_annotation_tags",
        ):
            if "task_identity" not in _table_columns(connection, table):
                raise DatabaseSchemaError(
                    f"viewer database schema is missing {table}.task_identity"
                )
        return

    task_tables = {
        "registered_tasks",
        "task_annotations",
        "task_annotation_tags",
    }.intersection(tables)
    if task_tables:
        legacy_tables = {
            table
            for table in task_tables
            if "task_identity" not in _table_columns(connection, table)
        }
        if legacy_tables != task_tables:
            raise DatabaseSchemaError(
                "viewer database has task tables without schema metadata"
            )
        populated = [
            table
            for table in ("task_annotations", "task_annotation_tags")
            if table in legacy_tables
            if connection.execute(f"SELECT count(*) FROM {table}").fetchone()
            != (0,)
        ]
        if populated:
            raise DatabaseSchemaError(
                "legacy task annotation rows lack authenticated "
                "task_identity and cannot be migrated: " + ", ".join(populated)
            )
        for table in (
            "task_annotation_tags",
            "task_annotations",
            "registered_tasks",
        ):
            if table in legacy_tables:
                connection.execute(f"DROP TABLE {table}")

    connection.execute(
        """
        CREATE TABLE viewer_schema (
            schema_version INTEGER PRIMARY KEY
        )
        """
    )
    connection.execute(
        "INSERT INTO viewer_schema(schema_version) VALUES (?)",
        [_SCHEMA_VERSION],
    )


def _table_columns(
    connection: duckdb.DuckDBPyConnection,
    table: str,
) -> set[str]:
    return {
        row[1]
        for row in connection.execute(
            f"PRAGMA table_info('{table}')"
        ).fetchall()
    }


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
