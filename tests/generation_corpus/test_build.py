from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

from dr_code.generation_corpus import build_generation_corpus
from dr_code.generation_corpus.models import SourceManifest
from dr_code.generation_corpus.writer import CorpusPopulation, CorpusWriter


@dataclass(frozen=True, slots=True)
class _EmptyAdapter:
    adapter_name: str = "empty_fixture"
    adapter_version: int = 3

    def populate(
        self,
        *,
        dump_directory: Path,
        source_manifest: SourceManifest,
        writer: CorpusWriter,
    ) -> None:
        assert dump_directory.is_dir()
        assert source_manifest.pools == ()
        assert writer is not None


def _write_manifest(directory: Path) -> None:
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "2026-06-21T00:00:00+00:00",
                "output_dir": "/archived/source",
                "pools": [],
            }
        ),
        encoding="utf-8",
    )


def test_builds_and_atomically_publishes_adapter_output(
    tmp_path: Path,
) -> None:
    dump_directory = tmp_path / "dump"
    dump_directory.mkdir()
    _write_manifest(dump_directory)
    destination = tmp_path / "corpus"

    manifest = build_generation_corpus(
        dump_directory=dump_directory,
        destination=destination,
        adapter=_EmptyAdapter(),
        expected_population=CorpusPopulation(
            generations=0,
            source_records=0,
            encoder_artifacts=0,
            requests=0,
            tasks=0,
        ),
        created_at="2026-08-08T00:00:00+00:00",
    )

    assert manifest.adapter_name == "empty_fixture"
    assert manifest.adapter_version == 3
    assert manifest.generations.rows == 0
    assert pl.read_parquet(destination / "generations.parquet").is_empty()
    assert (
        json.loads((destination / "manifest.json").read_text())["created_at"]
        == "2026-08-08T00:00:00+00:00"
    )


def test_failure_does_not_publish_destination(tmp_path: Path) -> None:
    dump_directory = tmp_path / "dump"
    dump_directory.mkdir()
    _write_manifest(dump_directory)
    destination = tmp_path / "corpus"

    @dataclass(frozen=True, slots=True)
    class FailingAdapter:
        adapter_name: str = "failing_fixture"
        adapter_version: int = 1

        def populate(
            self,
            *,
            dump_directory: Path,
            source_manifest: SourceManifest,
            writer: CorpusWriter,
        ) -> None:
            raise ValueError("fixture failure")

    try:
        build_generation_corpus(
            dump_directory=dump_directory,
            destination=destination,
            adapter=FailingAdapter(),
        )
    except ValueError as exc:
        assert str(exc) == "fixture failure"
    else:
        raise AssertionError("expected adapter failure")
    assert not destination.exists()


def test_population_mismatch_does_not_publish_destination(
    tmp_path: Path,
) -> None:
    dump_directory = tmp_path / "dump"
    dump_directory.mkdir()
    _write_manifest(dump_directory)
    destination = tmp_path / "corpus"

    with pytest.raises(ValueError, match="population differs"):
        build_generation_corpus(
            dump_directory=dump_directory,
            destination=destination,
            adapter=_EmptyAdapter(),
            expected_population=CorpusPopulation(
                generations=1,
                source_records=0,
                encoder_artifacts=0,
                requests=1,
                tasks=0,
            ),
        )

    assert not destination.exists()
