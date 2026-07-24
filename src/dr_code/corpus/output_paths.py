"""Preflight checks for output namespaces that must never follow symlinks."""

from __future__ import annotations

import os
from pathlib import Path


class UnsafeOutputPathError(ValueError):
    """An output path could redirect writes outside its owned namespace."""


def lexical_absolute(path: Path | str) -> Path:
    """Return an absolute path without resolving any symlink components."""

    return Path(os.path.abspath(Path(path).expanduser()))


def validate_output_path(path: Path | str, *, label: str) -> Path:
    """Reject existing symlink components and non-directory output roots."""

    absolute = lexical_absolute(path)
    _reject_symlink_components(absolute, label=label)
    if absolute.exists() and not absolute.is_dir():
        raise UnsafeOutputPathError(f"{label} must be a directory")
    return absolute


def validate_owned_tree(path: Path, *, label: str) -> None:
    """Reject every symlink below an existing owned output directory."""

    _reject_symlink_components(path, label=label)
    if not path.exists():
        return
    if not path.is_dir():
        raise UnsafeOutputPathError(f"{label} must be a directory")
    for directory, directory_names, filenames in os.walk(
        path, followlinks=False
    ):
        parent = Path(directory)
        for name in (*directory_names, *filenames):
            child = parent / name
            if child.is_symlink():
                raise UnsafeOutputPathError(
                    f"{label} must not contain symlinks: {child}"
                )


def validate_reserved_path(path: Path, *, label: str) -> None:
    """Reject a reserved path if it or an existing ancestor is a symlink."""

    _reject_symlink_components(path, label=label)


def _reject_symlink_components(path: Path, *, label: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise UnsafeOutputPathError(
                f"{label} must not use symlink aliases: {current}"
            )
        if not current.exists():
            break


__all__ = (
    "UnsafeOutputPathError",
    "lexical_absolute",
    "validate_output_path",
    "validate_owned_tree",
    "validate_reserved_path",
)
