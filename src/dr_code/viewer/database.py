"""DuckDB catalog, migrations, and mutable annotation persistence."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Final

import duckdb

from dr_code.viewer.domain import (
    Annotation,
    AnnotationOrigin,
    InvalidQueryError,
    RunDescriptor,
    Tag,
    TaskAnnotation,
    TaskAnnotationProvenance,
    TaskIdentity,
    Verdict,
    decode_task_provenance,
    encode_task_provenance,
    validate_sha256,
    validate_task_identity,
)


_MIGRATIONS: Final[tuple[tuple[int, tuple[str, ...]], ...]] = (
    (
        1,
        (
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
                verdict VARCHAR NOT NULL CHECK (
                    verdict IN ('should_be_parseable', 'expected_no_code')
                ),
                note VARCHAR,
                created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
                PRIMARY KEY (
                    corpus_sha256, sample_id, decoder_output_sha256
                )
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
        ),
    ),
    (
        2,
        ("ALTER TABLE annotations ALTER COLUMN verdict DROP NOT NULL",),
    ),
    (
        3,
        (
            """
            CREATE TABLE task_annotations (
                dataset_id VARCHAR NOT NULL,
                task_id VARCHAR NOT NULL,
                origin VARCHAR NOT NULL DEFAULT 'human' CHECK (
                    origin IN ('human', 'machine')
                ),
                category VARCHAR,
                note VARCHAR,
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
        ),
    ),
)


class ViewerDatabase:
    """Own the one-process DuckDB connection used by the viewer."""

    def __init__(self, path: str | Path) -> None:
        raw_path = str(path)
        if raw_path != ":memory:":
            database_path = Path(path).expanduser().resolve()
            database_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path = str(database_path)
        self._connection = duckdb.connect(raw_path)
        self._migrate()

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._connection

    def close(self) -> None:
        self._connection.close()

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

    def get_task_annotation(
        self, dataset_id: str, task_id: str
    ) -> TaskAnnotation | None:
        identity = validate_task_identity(dataset_id, task_id)
        row = self._connection.execute(
            """
            SELECT origin, category, note, provenance
            FROM task_annotations
            WHERE dataset_id = ? AND task_id = ?
            """,
            [identity.dataset_id, identity.task_id],
        ).fetchone()
        if row is None:
            return None
        tags = self._task_annotation_tags(identity)
        return TaskAnnotation(
            identity=identity,
            origin=AnnotationOrigin(row[0]),
            category=row[1],
            note=row[2],
            tags=tags,
            provenance=decode_task_provenance(row[3]),
        )

    def put_task_annotation(
        self,
        dataset_id: str,
        task_id: str,
        *,
        origin: AnnotationOrigin | str = AnnotationOrigin.HUMAN,
        category: str | None = None,
        note: str | None = None,
        tag_ids: Iterable[str] = (),
        provenance: TaskAnnotationProvenance | None = None,
    ) -> TaskAnnotation:
        identity = validate_task_identity(dataset_id, task_id)
        try:
            parsed_origin = AnnotationOrigin(origin)
        except ValueError as exc:
            raise InvalidQueryError(
                f"unsupported task annotation origin: {origin}"
            ) from exc
        normalized_note = _normalize_note(note)
        normalized_category = _normalize_category(category)
        provenance_json = encode_task_provenance(provenance)
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
                INSERT INTO task_annotations(
                    dataset_id, task_id, origin, category, note, provenance
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id, task_id) DO UPDATE SET
                    origin = excluded.origin,
                    category = excluded.category,
                    note = excluded.note,
                    provenance = excluded.provenance,
                    updated_at = now()
                """,
                [
                    identity.dataset_id,
                    identity.task_id,
                    parsed_origin.value,
                    normalized_category,
                    normalized_note,
                    provenance_json,
                ],
            )
            self._connection.execute(
                """
                DELETE FROM task_annotation_tags
                WHERE dataset_id = ? AND task_id = ?
                """,
                [identity.dataset_id, identity.task_id],
            )
            for tag_id in selected_tag_ids:
                self._connection.execute(
                    """
                    INSERT INTO task_annotation_tags(
                        dataset_id, task_id, tag_id
                    ) VALUES (?, ?, ?)
                    """,
                    [identity.dataset_id, identity.task_id, tag_id],
                )

        self._transaction(upsert)
        annotation = self.get_task_annotation(
            identity.dataset_id, identity.task_id
        )
        assert annotation is not None
        return annotation

    def delete_task_annotation(self, dataset_id: str, task_id: str) -> bool:
        identity = validate_task_identity(dataset_id, task_id)
        deleted = False

        def remove() -> None:
            nonlocal deleted
            self._connection.execute(
                """
                DELETE FROM task_annotation_tags
                WHERE dataset_id = ? AND task_id = ?
                """,
                [identity.dataset_id, identity.task_id],
            )
            result = self._connection.execute(
                """
                DELETE FROM task_annotations
                WHERE dataset_id = ? AND task_id = ?
                RETURNING task_id
                """,
                [identity.dataset_id, identity.task_id],
            ).fetchone()
            deleted = result is not None

        self._transaction(remove)
        return deleted

    def export_task_annotations(self) -> list[dict[str, object]]:
        """Return a timestamp- and machine-path-free deterministic export."""
        rows = self._connection.execute(
            """
            SELECT
                a.dataset_id,
                a.task_id,
                a.origin,
                a.category,
                a.note,
                a.provenance,
                coalesce(
                    list(t.name ORDER BY t.normalized_name, t.tag_id)
                        FILTER (WHERE t.tag_id IS NOT NULL),
                    []
                ) AS tags
            FROM task_annotations AS a
            LEFT JOIN task_annotation_tags AS atag USING (dataset_id, task_id)
            LEFT JOIN tags AS t USING (tag_id)
            GROUP BY
                a.dataset_id,
                a.task_id,
                a.origin,
                a.category,
                a.note,
                a.provenance
            ORDER BY a.dataset_id, a.task_id
            """
        ).fetchall()
        return [
            {
                "dataset_id": row[0],
                "task_id": row[1],
                "origin": row[2],
                "category": row[3],
                "note": row[4],
                "provenance": row[5],
                "tags": row[6],
            }
            for row in rows
        ]

    def applied_migrations(self) -> tuple[int, ...]:
        rows = self._connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        return tuple(row[0] for row in rows)

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

    def _task_annotation_tags(
        self, identity: TaskIdentity
    ) -> tuple[Tag, ...]:
        rows = self._connection.execute(
            """
            SELECT t.tag_id, t.name
            FROM task_annotation_tags AS atag
            JOIN tags AS t USING (tag_id)
            WHERE atag.dataset_id = ? AND atag.task_id = ?
            ORDER BY t.normalized_name, t.tag_id
            """,
            [identity.dataset_id, identity.task_id],
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

    def _migrate(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
            )
            """
        )
        applied = set(self.applied_migrations())
        known = {version for version, _ in _MIGRATIONS}
        unknown = sorted(applied.difference(known))
        if unknown:
            raise RuntimeError(
                "database has unsupported schema migration(s): "
                + ", ".join(map(str, unknown))
            )
        for version, statements in _MIGRATIONS:
            if version in applied:
                continue

            def apply() -> None:
                for statement in statements:
                    self._connection.execute(statement)
                self._connection.execute(
                    "INSERT INTO schema_migrations(version) VALUES (?)",
                    [version],
                )

            self._transaction(apply)

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


def _normalize_category(category: str | None) -> str | None:
    if category is None:
        return None
    if not isinstance(category, str):
        raise InvalidQueryError(
            "task annotation category must be a string or null"
        )
    normalized = category.strip()
    if not normalized:
        return None
    if len(normalized) > 256:
        raise InvalidQueryError(
            "task annotation category must be at most 256 characters"
        )
    return normalized


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


__all__ = ("ViewerDatabase",)
