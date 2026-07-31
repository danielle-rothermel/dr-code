"""Concurrency-safe publication of immutable output directories."""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from dr_code.corpus.durability import fsync_directory, fsync_file

_MACOS_RENAME_EXCL = 0x00000004
_LINUX_AT_FDCWD = -100
_LINUX_RENAME_NOREPLACE = 0x1


class AtomicPublicationError(OSError):
    """The platform cannot guarantee atomic no-replace publication."""


@contextmanager
def staged_output_directory(destination: Path) -> Iterator[Path]:
    """Yield a private sibling directory and publish it without replacement."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    try:
        yield temporary
        publish_staged_output_directory(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _publish_without_replacement(temporary: Path, destination: Path) -> None:
    source_bytes = os.fsencode(temporary)
    destination_bytes = os.fsencode(destination)
    if b"\0" in source_bytes or b"\0" in destination_bytes:
        raise AtomicPublicationError(
            "atomic publication paths cannot contain null bytes"
        )
    system = platform.system()
    library = ctypes.CDLL(None, use_errno=True)
    if system == "Darwin":
        result = _macos_exclusive_rename(
            library,
            source_bytes,
            destination_bytes,
        )
    elif system == "Linux":
        result = _linux_exclusive_rename(
            library,
            source_bytes,
            destination_bytes,
        )
    else:
        raise AtomicPublicationError(
            f"atomic no-replace publication is unsupported on {system!r}"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number,
            f"output already exists: {destination}",
            destination,
        )
    raise AtomicPublicationError(
        error_number,
        f"atomic no-replace publication failed for {destination}: "
        f"{os.strerror(error_number)}",
        destination,
    )


def publish_staged_output_directory(
    temporary: Path,
    destination: Path,
) -> None:
    """Publish an existing private sibling directory without replacement."""

    if temporary.parent.resolve() != destination.parent.resolve():
        raise AtomicPublicationError(
            "atomic publication requires source and destination siblings"
        )
    _fsync_staged_tree(temporary)
    _publish_without_replacement(temporary, destination)
    fsync_directory(destination.parent)


def _fsync_staged_tree(root: Path) -> None:
    """Make a completed private tree durable before it becomes visible."""

    regular_files: list[Path] = []
    directories: list[Path] = []
    for current, directory_names, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        directory_names.sort()
        current_path = Path(current)
        directories.append(current_path)
        for filename in sorted(filenames):
            path = current_path / filename
            if stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
                regular_files.append(path)

    for path in sorted(regular_files):
        fsync_file(path)
    for path in sorted(
        directories,
        key=lambda candidate: (-len(candidate.parts), str(candidate)),
    ):
        fsync_directory(path)


def _macos_exclusive_rename(
    library: ctypes.CDLL,
    source: bytes,
    destination: bytes,
) -> int:
    try:
        rename = library.renamex_np
    except AttributeError as exc:
        raise AtomicPublicationError(
            "macOS does not expose renamex_np for atomic publication"
        ) from exc
    rename.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    return rename(source, destination, _MACOS_RENAME_EXCL)


def _linux_exclusive_rename(
    library: ctypes.CDLL,
    source: bytes,
    destination: bytes,
) -> int:
    try:
        rename = library.renameat2
    except AttributeError as exc:
        raise AtomicPublicationError(
            "Linux libc does not expose renameat2 for atomic publication"
        ) from exc
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    return rename(
        _LINUX_AT_FDCWD,
        source,
        _LINUX_AT_FDCWD,
        destination,
        _LINUX_RENAME_NOREPLACE,
    )


__all__ = (
    "AtomicPublicationError",
    "publish_staged_output_directory",
    "staged_output_directory",
)
