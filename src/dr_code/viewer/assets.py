"""Validate and materialize the immutable preprocessing-viewer frontend."""

from __future__ import annotations

import fcntl
import hashlib
import io
import os
import shutil
import stat
import sys
import tarfile
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Final, NamedTuple

ARCHIVE_FILENAME: Final = "prebuilt_viewer_assets.tar.gz"
DIGEST_FILENAME: Final = "prebuilt_viewer_assets.sha256"
_CACHE_MARKER_FILENAME: Final = ".archive.sha256"
_CACHE_LOCKS_GUARD: Final = threading.Lock()
_CACHE_LOCKS: Final[dict[Path, threading.Lock]] = {}


class _TreeEntry(NamedTuple):
    kind: str
    mode: int
    size: int
    sha256: str | None


class PrebuiltViewerAssetsError(RuntimeError):
    """The prebuilt viewer archive or its extracted contents are invalid."""


def validate_static_directory(static_dir: Path) -> None:
    """Validate the minimum immutable frontend contract."""
    if not (static_dir / "index.html").is_file():
        raise PrebuiltViewerAssetsError(
            "prebuilt preprocessing-viewer archive has no index.html"
        )
    assets = static_dir / "assets"
    if not assets.is_dir() or not any(
        path.is_file() for path in assets.iterdir()
    ):
        raise PrebuiltViewerAssetsError(
            "prebuilt preprocessing-viewer archive has no assets"
        )
    if any(path.is_symlink() for path in static_dir.rglob("*")):
        raise PrebuiltViewerAssetsError(
            "prebuilt preprocessing-viewer assets contain a symlink"
        )


def extract_prebuilt_viewer_archive(
    archive_path: Path,
    digest_path: Path,
    destination: Path,
) -> Path:
    """Digest-check and safely extract the archive below ``destination``."""
    archive_bytes, _digest = _read_verified_archive(archive_path, digest_path)
    expected = _expected_tree_manifest(archive_bytes)
    static_dir = _extract_archive_bytes(archive_bytes, destination)
    validate_static_directory(static_dir)
    if not _tree_matches_manifest(static_dir, expected):
        raise PrebuiltViewerAssetsError(
            "prebuilt preprocessing-viewer extraction is not authentic"
        )
    return static_dir


def materialize_prebuilt_viewer_assets(
    package_dir: Path,
    *,
    cache_root: Path | None = None,
) -> Path:
    """Return an atomically cached extraction of the packaged archive."""
    archive_path = package_dir / ARCHIVE_FILENAME
    digest_path = package_dir / DIGEST_FILENAME
    archive_bytes, digest = _read_verified_archive(archive_path, digest_path)
    expected = _expected_tree_manifest(archive_bytes)
    root = (
        cache_root.expanduser().resolve()
        if cache_root is not None
        else _default_cache_root()
    )
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PrebuiltViewerAssetsError(
            f"cannot create preprocessing-viewer asset cache: {root}"
        ) from exc

    cached = root / digest
    lock_path = root / f".{digest}.lock"
    with _cache_lock(lock_path):
        static_dir = cached / "static"
        if _cached_assets_are_valid(cached, digest, expected):
            return static_dir
        _remove_cache_entry(cached)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{digest}.", dir=root)
        ).resolve()
        try:
            extracted = _extract_archive_bytes(archive_bytes, staging)
            validate_static_directory(extracted)
            if not _tree_matches_manifest(extracted, expected):
                raise PrebuiltViewerAssetsError(
                    "prebuilt preprocessing-viewer extraction is not authentic"
                )
            (staging / _CACHE_MARKER_FILENAME).write_text(
                digest + "\n", encoding="ascii"
            )
            os.replace(staging, cached)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return cached / "static"


def _read_verified_archive(
    archive_path: Path,
    digest_path: Path,
) -> tuple[bytes, str]:
    try:
        recorded = digest_path.read_text(encoding="ascii").strip()
        archive_bytes = archive_path.read_bytes()
    except (OSError, UnicodeError) as exc:
        raise PrebuiltViewerAssetsError(
            "prebuilt preprocessing-viewer assets are missing"
        ) from exc
    actual = hashlib.sha256(archive_bytes).hexdigest()
    if (
        len(recorded) != 64
        or any(character not in "0123456789abcdef" for character in recorded)
        or recorded != actual
    ):
        raise PrebuiltViewerAssetsError(
            "prebuilt preprocessing-viewer asset digest is invalid"
        )
    return archive_bytes, actual


def _extract_archive_bytes(archive_bytes: bytes, destination: Path) -> Path:
    try:
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(
            fileobj=io.BytesIO(archive_bytes), mode="r:gz"
        ) as archive:
            members = archive.getmembers()
            _validate_archive_members(members)
            for member in members:
                _extract_member(archive, member, destination)
    except PrebuiltViewerAssetsError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise PrebuiltViewerAssetsError(
            "prebuilt preprocessing-viewer archive cannot be extracted"
        ) from exc
    return destination / "static"


def _expected_tree_manifest(
    archive_bytes: bytes,
) -> dict[PurePosixPath, _TreeEntry]:
    expected = {
        PurePosixPath("."): _TreeEntry(
            kind="directory", mode=0o755, size=0, sha256=None
        )
    }

    def add_directory(path: PurePosixPath) -> None:
        existing = expected.get(path)
        entry = _TreeEntry(kind="directory", mode=0o755, size=0, sha256=None)
        if existing is not None and existing != entry:
            raise PrebuiltViewerAssetsError(
                "prebuilt preprocessing-viewer archive is unsafe"
            )
        expected[path] = entry

    try:
        with tarfile.open(
            fileobj=io.BytesIO(archive_bytes), mode="r:gz"
        ) as archive:
            members = archive.getmembers()
            _validate_archive_members(members)
            for member in members:
                archive_path = PurePosixPath(member.name)
                relative_parts = archive_path.parts[1:]
                relative = (
                    PurePosixPath(*relative_parts)
                    if relative_parts
                    else PurePosixPath(".")
                )
                for parent in reversed(relative.parents):
                    add_directory(parent)
                if member.isdir():
                    add_directory(relative)
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise PrebuiltViewerAssetsError(
                        "prebuilt preprocessing-viewer archive is malformed"
                    )
                with source:
                    content = source.read()
                if len(content) != member.size:
                    raise PrebuiltViewerAssetsError(
                        "prebuilt preprocessing-viewer archive is malformed"
                    )
                if relative in expected:
                    raise PrebuiltViewerAssetsError(
                        "prebuilt preprocessing-viewer archive is unsafe"
                    )
                expected[relative] = _TreeEntry(
                    kind="file",
                    mode=0o644,
                    size=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                )
    except PrebuiltViewerAssetsError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise PrebuiltViewerAssetsError(
            "prebuilt preprocessing-viewer archive cannot be inspected"
        ) from exc
    return expected


def _validate_archive_members(members: list[tarfile.TarInfo]) -> None:
    names: set[str] = set()
    for member in members:
        path = PurePosixPath(member.name)
        canonical_name = path.as_posix()
        if (
            canonical_name in names
            or path.is_absolute()
            or not path.parts
            or path.parts[0] != "static"
            or ".." in path.parts
            or not (member.isfile() or member.isdir())
        ):
            raise PrebuiltViewerAssetsError(
                "prebuilt preprocessing-viewer archive is unsafe"
            )
        names.add(canonical_name)


def _extract_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: Path,
) -> None:
    relative = PurePosixPath(member.name)
    target = destination.joinpath(*relative.parts)
    if member.isdir():
        target.mkdir(parents=True, exist_ok=True)
        target.chmod(0o755)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    source = archive.extractfile(member)
    if source is None:
        raise PrebuiltViewerAssetsError(
            "prebuilt preprocessing-viewer archive is malformed"
        )
    with source, target.open("xb") as output:
        shutil.copyfileobj(source, output)
    target.chmod(0o644)


def _default_cache_root() -> Path:
    configured = os.environ.get("XDG_CACHE_HOME")
    if configured:
        base = Path(configured).expanduser()
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path.home() / ".cache"
    return (base / "dr-code" / "viewer-assets").resolve()


@contextmanager
def _cache_lock(lock_path: Path) -> Iterator[None]:
    with _CACHE_LOCKS_GUARD:
        thread_lock = _CACHE_LOCKS.setdefault(lock_path, threading.Lock())
    with thread_lock:
        try:
            stream = lock_path.open("a+b")
        except OSError as exc:
            raise PrebuiltViewerAssetsError(
                f"cannot open preprocessing-viewer cache lock: {lock_path}"
            ) from exc
        try:
            _lock_stream(stream)
            yield
        finally:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()


def _lock_stream(stream: BinaryIO) -> None:
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
    except OSError as exc:
        raise PrebuiltViewerAssetsError(
            "cannot lock preprocessing-viewer asset cache"
        ) from exc


def _cached_assets_are_valid(
    cache_dir: Path,
    digest: str,
    expected: dict[PurePosixPath, _TreeEntry],
) -> bool:
    try:
        marker = (cache_dir / _CACHE_MARKER_FILENAME).read_text(
            encoding="ascii"
        )
        if marker.strip() != digest:
            return False
        static_dir = cache_dir / "static"
        validate_static_directory(static_dir)
        return _tree_matches_manifest(static_dir, expected)
    except (OSError, UnicodeError, PrebuiltViewerAssetsError):
        return False


def _tree_matches_manifest(
    static_dir: Path,
    expected: dict[PurePosixPath, _TreeEntry],
) -> bool:
    try:
        root_stat = static_dir.lstat()
        if not stat.S_ISDIR(root_stat.st_mode):
            return False
        actual = {
            PurePosixPath("."): _TreeEntry(
                kind="directory",
                mode=stat.S_IMODE(root_stat.st_mode),
                size=0,
                sha256=None,
            )
        }
        pending = [(static_dir, PurePosixPath("."))]
        while pending:
            directory, relative_directory = pending.pop()
            with os.scandir(directory) as entries:
                for item in entries:
                    relative = relative_directory / item.name
                    item_stat = item.stat(follow_symlinks=False)
                    mode = stat.S_IMODE(item_stat.st_mode)
                    if stat.S_ISDIR(item_stat.st_mode):
                        actual[relative] = _TreeEntry(
                            kind="directory",
                            mode=mode,
                            size=0,
                            sha256=None,
                        )
                        pending.append((Path(item.path), relative))
                    elif stat.S_ISREG(item_stat.st_mode):
                        content_digest = hashlib.sha256()
                        with Path(item.path).open("rb") as stream:
                            for chunk in iter(
                                lambda: stream.read(1024 * 1024), b""
                            ):
                                content_digest.update(chunk)
                        actual[relative] = _TreeEntry(
                            kind="file",
                            mode=mode,
                            size=item_stat.st_size,
                            sha256=content_digest.hexdigest(),
                        )
                    else:
                        return False
        return actual == expected
    except OSError:
        return False


def _remove_cache_entry(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


__all__ = (
    "ARCHIVE_FILENAME",
    "DIGEST_FILENAME",
    "PrebuiltViewerAssetsError",
    "extract_prebuilt_viewer_archive",
    "materialize_prebuilt_viewer_assets",
    "validate_static_directory",
)
