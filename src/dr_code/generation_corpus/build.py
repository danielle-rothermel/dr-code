from __future__ import annotations

from pathlib import Path

from dr_code.generation_corpus.adapters.base import CorpusAdapter
from dr_code.generation_corpus.models import BuildManifest
from dr_code.generation_corpus.pool_dump import read_manifest
from dr_code.generation_corpus.writer import CorpusPopulation, CorpusWriter


def build_generation_corpus(
    *,
    dump_directory: Path,
    destination: Path,
    adapter: CorpusAdapter,
    expected_population: CorpusPopulation | None = None,
    created_at: str | None = None,
) -> BuildManifest:
    """Build and atomically publish one validated generation corpus."""
    source_manifest_path = dump_directory / "manifest.json"
    source_manifest = read_manifest(source_manifest_path)
    writer = CorpusWriter(
        destination,
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        adapter_name=adapter.adapter_name,
        adapter_version=adapter.adapter_version,
        created_at=created_at,
    )
    try:
        adapter.populate(
            dump_directory=dump_directory,
            source_manifest=source_manifest,
            writer=writer,
        )
        if (
            expected_population is not None
            and writer.population != expected_population
        ):
            raise ValueError(
                "extracted corpus population differs from the audited "
                f"snapshot: expected={expected_population!r}, "
                f"actual={writer.population!r}"
            )
        return writer.publish()
    except BaseException:
        writer.abort()
        raise


__all__ = ["build_generation_corpus"]
