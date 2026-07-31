from __future__ import annotations

from pathlib import Path

import pytest

from dr_code.corpus.atomic_directory import (
    AtomicPublicationError,
    staged_output_directory,
)


def test_publication_has_one_exact_durability_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.corpus import atomic_directory

    destination = tmp_path / "output"
    events: list[str] = []
    original_file = atomic_directory.fsync_file
    original_directory = atomic_directory.fsync_directory
    original_rename = atomic_directory._publish_without_replacement

    def label(path: Path) -> str:
        relative = path.relative_to(tmp_path)
        return relative.as_posix() if relative.parts else "."

    def fsync_file(path: Path) -> None:
        events.append(f"file:{label(path)}")
        original_file(path)

    def fsync_directory(path: Path) -> None:
        events.append(f"directory:{label(path)}")
        original_directory(path)

    def rename(source: Path, target: Path) -> None:
        events.append(f"rename:{label(source)}->{label(target)}")
        original_rename(source, target)

    monkeypatch.setattr(atomic_directory, "fsync_file", fsync_file)
    monkeypatch.setattr(
        atomic_directory,
        "fsync_directory",
        fsync_directory,
    )
    monkeypatch.setattr(
        atomic_directory,
        "_publish_without_replacement",
        rename,
    )

    with staged_output_directory(destination) as temporary:
        temporary_name = temporary.name
        (temporary / "a.txt").write_text("a", encoding="utf-8")
        nested = temporary / "nested"
        nested.mkdir()
        (nested / "b.txt").write_text("b", encoding="utf-8")

    assert events == [
        f"file:{temporary_name}/a.txt",
        f"file:{temporary_name}/nested/b.txt",
        f"directory:{temporary_name}/nested",
        f"directory:{temporary_name}",
        f"rename:{temporary_name}->output",
        "directory:.",
    ]


@pytest.mark.parametrize(
    "failure_point",
    ("file", "staging-directory", "rename"),
)
def test_failure_before_rename_cleans_staging_without_visible_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    from dr_code.corpus import atomic_directory

    destination = tmp_path / "output"
    original_file = atomic_directory.fsync_file
    original_directory = atomic_directory.fsync_directory
    original_rename = atomic_directory._publish_without_replacement
    owned: Path

    def fsync_file(path: Path) -> None:
        if failure_point == "file":
            raise OSError("file fsync failed")
        original_file(path)

    def fsync_directory(path: Path) -> None:
        if failure_point == "staging-directory" and path == owned:
            raise OSError("staging directory fsync failed")
        original_directory(path)

    def rename(source: Path, target: Path) -> None:
        if failure_point == "rename":
            raise OSError("rename failed")
        original_rename(source, target)

    monkeypatch.setattr(atomic_directory, "fsync_file", fsync_file)
    monkeypatch.setattr(
        atomic_directory,
        "fsync_directory",
        fsync_directory,
    )
    monkeypatch.setattr(
        atomic_directory,
        "_publish_without_replacement",
        rename,
    )

    with pytest.raises(OSError, match="failed"):
        with staged_output_directory(destination) as temporary:
            owned = temporary
            (temporary / "complete.txt").write_text(
                "complete",
                encoding="utf-8",
            )

    assert not destination.exists()
    assert not owned.exists()
    assert not list(tmp_path.glob(".output.*.tmp"))


def test_parent_fsync_failure_reports_error_but_leaves_complete_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dr_code.corpus import atomic_directory

    destination = tmp_path / "output"
    original_directory = atomic_directory.fsync_directory

    def fsync_directory(path: Path) -> None:
        if path == destination.parent:
            raise OSError("parent fsync failed")
        original_directory(path)

    monkeypatch.setattr(
        atomic_directory,
        "fsync_directory",
        fsync_directory,
    )

    with pytest.raises(OSError, match="parent fsync failed"):
        with staged_output_directory(destination) as temporary:
            owned = temporary
            (temporary / "complete.txt").write_text(
                "complete",
                encoding="utf-8",
            )

    assert (destination / "complete.txt").read_text(encoding="utf-8") == (
        "complete"
    )
    assert not owned.exists()
    assert not list(tmp_path.glob(".output.*.tmp"))


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
