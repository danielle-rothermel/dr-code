from __future__ import annotations

from pathlib import Path

import pytest

from dr_code.corpus.atomic_directory import (
    AtomicPublicationError,
    staged_output_directory,
)


def test_publication_never_replaces_destination_created_at_boundary(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "output"

    with pytest.raises(FileExistsError, match="output already exists"):
        with staged_output_directory(destination) as temporary:
            (temporary / "candidate").write_text("candidate", encoding="utf-8")
            destination.mkdir()

    assert list(destination.iterdir()) == []
    assert not list(tmp_path.glob(".output.*.tmp"))


def test_abandoned_staging_does_not_block_later_owner(
    tmp_path: Path,
) -> None:
    abandoned = tmp_path / ".output.abandoned.tmp"
    abandoned.mkdir()
    (abandoned / "partial").write_text("partial", encoding="utf-8")
    destination = tmp_path / "output"

    with staged_output_directory(destination) as temporary:
        owned = temporary
        (temporary / "complete").write_text("complete", encoding="utf-8")

    assert (destination / "complete").read_text(encoding="utf-8") == "complete"
    assert (abandoned / "partial").read_text(encoding="utf-8") == "partial"
    assert not owned.exists()


def test_unsupported_platform_fails_closed_and_cleans_owned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dr_code.corpus.atomic_directory.platform.system", lambda: "Plan9"
    )
    destination = tmp_path / "output"

    with pytest.raises(AtomicPublicationError, match="unsupported"):
        with staged_output_directory(destination) as temporary:
            owned = temporary
            (temporary / "candidate").write_text("candidate", encoding="utf-8")

    assert not owned.exists()
    assert not destination.exists()
