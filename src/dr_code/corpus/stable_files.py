"""Disk-backed snapshots that bind validation and consumption to one byte set."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_COPY_BUFFER_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class StableFile:
    """One immutable-for-the-operation file copy and its authenticated bytes."""

    source_path: Path
    path: Path
    sha256: str
    size: int

    def descriptor(self) -> dict[str, object]:
        return {
            "sha256": self.sha256,
            "size": self.size,
        }


@contextmanager
def stable_file(
    path: Path | str,
    *,
    label: str = "input",
    max_bytes: int | None = None,
) -> Iterator[StableFile]:
    """Capture a regular file once, then expose only the captured bytes.

    ``max_bytes`` bounds unauthenticated capture before a caller parses or
    otherwise consumes the snapshot.  It is enforced while streaming, rather
    than from source metadata, so a concurrent file change cannot bypass it.
    """

    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"{label} must be a regular file: {source}")
    with tempfile.TemporaryDirectory(prefix="dr-code-stable-file-") as root:
        destination = Path(root) / "snapshot"
        yield _copy_and_hash(
            source,
            destination,
            label=label,
            max_bytes=max_bytes,
        )


@contextmanager
def stable_files(
    paths: Mapping[str, Path | str],
) -> Iterator[dict[str, StableFile]]:
    """Capture a named file set before any caller validation or processing."""

    with tempfile.TemporaryDirectory(prefix="dr-code-stable-files-") as root:
        snapshots: dict[str, StableFile] = {}
        for index, (label, raw_path) in enumerate(paths.items()):
            source = Path(raw_path).expanduser().resolve(strict=True)
            if not source.is_file():
                raise ValueError(f"{label} must be a regular file: {source}")
            snapshots[label] = _copy_and_hash(
                source,
                Path(root) / f"{index:04d}",
                label=label,
            )
        yield snapshots


def _copy_and_hash(
    source: Path,
    destination: Path,
    *,
    label: str,
    max_bytes: int | None = None,
) -> StableFile:
    if max_bytes is not None and (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 0
    ):
        raise ValueError("max_bytes must be a non-negative integer")
    digest = hashlib.sha256()
    size = 0
    try:
        with (
            source.open("rb") as input_stream,
            destination.open("xb") as output_stream,
        ):
            while chunk := input_stream.read(
                _COPY_BUFFER_SIZE
                if max_bytes is None
                else min(_COPY_BUFFER_SIZE, max_bytes - size + 1)
            ):
                if max_bytes is not None and size + len(chunk) > max_bytes:
                    raise ValueError(
                        f"{label} exceeds maximum size of {max_bytes} bytes"
                    )
                digest.update(chunk)
                output_stream.write(chunk)
                size += len(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise ValueError(f"cannot capture stable {label}: {source}") from exc
    return StableFile(
        source_path=source,
        path=destination,
        sha256=digest.hexdigest(),
        size=size,
    )


__all__ = ["StableFile", "stable_file", "stable_files"]
