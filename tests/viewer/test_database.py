from __future__ import annotations

import json

import duckdb
import pytest

from dr_code.viewer.database import ViewerDatabase
from dr_code.viewer.domain import InvalidQueryError, Verdict


CORPUS_HASH = "a" * 64
OUTPUT_HASH = "b" * 64


def test_migrations_and_annotations_survive_restart(tmp_path) -> None:
    path = tmp_path / "viewer.duckdb"
    with ViewerDatabase(path) as database:
        tag = database.create_tag("  Needs   Import ")
        duplicate = database.create_tag("needs import")
        annotation = database.put_annotation(
            CORPUS_HASH,
            "sample-1",
            OUTPUT_HASH,
            verdict=Verdict.SHOULD_BE_PARSEABLE,
            note="look again",
            tag_ids=[tag.tag_id],
        )
        assert database.applied_migrations() == (1, 2)
        assert duplicate == tag
        assert annotation.tags == (tag,)

    with ViewerDatabase(path) as reopened:
        actual = reopened.get_annotation(
            CORPUS_HASH, "sample-1", OUTPUT_HASH
        )

    assert actual is not None
    assert actual.verdict is Verdict.SHOULD_BE_PARSEABLE
    assert actual.note == "look again"
    assert actual.tags[0].name == "Needs Import"


def test_v1_migration_preserves_data_and_verdict_check(tmp_path) -> None:
    path = tmp_path / "viewer-v1.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute(
        """
        CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
        );
        INSERT INTO schema_migrations(version) VALUES (1);
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
            PRIMARY KEY (corpus_sha256, sample_id, decoder_output_sha256)
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
        );
        INSERT INTO tags VALUES ('tag-v1', 'Existing', 'existing', now());
        """
    )
    connection.execute(
        """
        INSERT INTO annotations VALUES (
            ?, 'sample-v1', ?, 'should_be_parseable', 'preserved', now(), now()
        )
        """,
        [CORPUS_HASH, OUTPUT_HASH],
    )
    connection.execute(
        """
        INSERT INTO annotation_tags VALUES (?, 'sample-v1', ?, 'tag-v1')
        """,
        [CORPUS_HASH, OUTPUT_HASH],
    )
    connection.close()

    with ViewerDatabase(path) as database:
        actual = database.get_annotation(
            CORPUS_HASH, "sample-v1", OUTPUT_HASH
        )
        assert database.applied_migrations() == (1, 2)
        assert actual is not None
        assert actual.verdict is Verdict.SHOULD_BE_PARSEABLE
        assert actual.note == "preserved"
        assert [tag.name for tag in actual.tags] == ["Existing"]

        database.put_annotation(
            CORPUS_HASH,
            "sample-v1",
            OUTPUT_HASH,
            verdict=None,
            note="still here",
            tag_ids=["tag-v1"],
        )
        with pytest.raises(duckdb.ConstraintException, match="CHECK"):
            database.connection.execute(
                """
                INSERT INTO annotations(
                    corpus_sha256, sample_id, decoder_output_sha256, verdict
                ) VALUES (?, 'invalid', ?, 'not-a-verdict')
                """,
                [CORPUS_HASH, "c" * 64],
            )


def test_unlabeled_note_and_tags_survive_restart(tmp_path) -> None:
    path = tmp_path / "viewer.duckdb"
    with ViewerDatabase(path) as database:
        tag = database.create_tag("Needs discussion")
        database.put_annotation(
            CORPUS_HASH,
            "sample-1",
            OUTPUT_HASH,
            verdict=None,
            note="keep without a verdict",
            tag_ids=[tag.tag_id],
        )

    with ViewerDatabase(path) as reopened:
        actual = reopened.get_annotation(
            CORPUS_HASH, "sample-1", OUTPUT_HASH
        )

    assert actual is not None
    assert actual.verdict is None
    assert actual.note == "keep without a verdict"
    assert [tag.name for tag in actual.tags] == ["Needs discussion"]


def test_annotation_output_hash_is_part_of_identity() -> None:
    with ViewerDatabase(":memory:") as database:
        database.put_annotation(
            CORPUS_HASH,
            "sample-1",
            OUTPUT_HASH,
            verdict=Verdict.EXPECTED_NO_CODE,
            note=None,
        )

        assert (
            database.get_annotation(CORPUS_HASH, "sample-1", "c" * 64)
            is None
        )


def test_annotation_export_is_deterministic_and_machine_independent() -> None:
    with ViewerDatabase(":memory:") as database:
        second = database.create_tag("z-last")
        first = database.create_tag("A-first")
        database.put_annotation(
            "f" * 64,
            "sample-z",
            "e" * 64,
            verdict=Verdict.EXPECTED_NO_CODE,
            note="",
            tag_ids=[second.tag_id, first.tag_id],
        )
        database.put_annotation(
            CORPUS_HASH,
            "sample-a",
            OUTPUT_HASH,
            verdict=Verdict.SHOULD_BE_PARSEABLE,
            note=None,
        )

        first_export = database.export_annotations()
        second_export = database.export_annotations()

    assert first_export == second_export
    assert [row["sample_id"] for row in first_export] == [
        "sample-a",
        "sample-z",
    ]
    assert first_export[1]["tags"] == ["A-first", "z-last"]
    assert "created_at" not in json.dumps(first_export)


def test_annotation_export_is_deterministic_with_null_verdict() -> None:
    with ViewerDatabase(":memory:") as database:
        tag = database.create_tag("review later")
        database.put_annotation(
            CORPUS_HASH,
            "sample-1",
            OUTPUT_HASH,
            verdict=None,
            note="unlabeled",
            tag_ids=[tag.tag_id],
        )

        first = database.export_annotations()
        second = database.export_annotations()

    assert first == second == [
        {
            "corpus_sha256": CORPUS_HASH,
            "sample_id": "sample-1",
            "decoder_output_sha256": OUTPUT_HASH,
            "verdict": None,
            "note": "unlabeled",
            "tags": ["review later"],
        }
    ]


def test_annotation_upsert_rejects_unknown_tag_atomically() -> None:
    with ViewerDatabase(":memory:") as database:
        with pytest.raises(InvalidQueryError, match="unknown tag"):
            database.put_annotation(
                CORPUS_HASH,
                "sample-1",
                OUTPUT_HASH,
                verdict=Verdict.SHOULD_BE_PARSEABLE,
                note=None,
                tag_ids=["missing"],
            )

        assert (
            database.get_annotation(CORPUS_HASH, "sample-1", OUTPUT_HASH)
            is None
        )


def test_delete_annotation_removes_tag_links() -> None:
    with ViewerDatabase(":memory:") as database:
        tag = database.create_tag("review")
        database.put_annotation(
            CORPUS_HASH,
            "sample-1",
            OUTPUT_HASH,
            verdict=Verdict.SHOULD_BE_PARSEABLE,
            note=None,
            tag_ids=[tag.tag_id],
        )

        assert database.delete_annotation(CORPUS_HASH, "sample-1", OUTPUT_HASH)
        assert not database.delete_annotation(
            CORPUS_HASH, "sample-1", OUTPUT_HASH
        )
