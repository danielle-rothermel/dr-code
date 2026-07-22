from __future__ import annotations

import json

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
        assert database.applied_migrations() == (1,)
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
