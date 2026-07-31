"""Filesystem durability primitives for atomic artifact publication."""

from __future__ import annotations

import os
from pathlib import Path


def fsync_file(path: Path) -> None:
    """Flush one completed regular file to durable storage."""

    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def fsync_directory(path: Path) -> None:
    """Flush directory-entry changes to durable storage."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ("fsync_directory", "fsync_file")
