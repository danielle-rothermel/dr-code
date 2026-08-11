from __future__ import annotations

from pathlib import Path
from typing import Protocol

from dr_code.generation_corpus.models import SourceManifest
from dr_code.generation_corpus.writer import CorpusWriter


class CorpusAdapter(Protocol):
    """Corpus adapter protocol: populate one historical dataset's validated tables."""

    adapter_name: str
    adapter_version: int

    def populate(
        self,
        *,
        dump_directory: Path,
        source_manifest: SourceManifest,
        writer: CorpusWriter,
    ) -> None: ...


__all__ = ["CorpusAdapter"]
