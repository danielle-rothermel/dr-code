from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import dr_code.viewer.assets as viewer_assets
from dr_code.viewer.assets import (
    PrebuiltViewerAssetsError,
    extract_prebuilt_viewer_archive,
    materialize_prebuilt_viewer_assets,
)


def test_materialization_is_atomic_under_high_thread_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package"
    _write_archive(
        package,
        {
            "static/index.html": b"<main>viewer</main>",
            "static/assets/app.js": b"console.log('viewer')",
        },
    )
    cache = tmp_path / "cache"
    workers = 24
    start = threading.Barrier(workers)
    extraction_count = 0
    extraction_guard = threading.Lock()
    real_extract = viewer_assets._extract_archive_bytes

    def counted_extract(archive_bytes: bytes, destination: Path) -> Path:
        nonlocal extraction_count
        with extraction_guard:
            extraction_count += 1
        return real_extract(archive_bytes, destination)

    monkeypatch.setattr(
        viewer_assets, "_extract_archive_bytes", counted_extract
    )

    def materialize(_index: int) -> Path:
        start.wait()
        return materialize_prebuilt_viewer_assets(package, cache_root=cache)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(materialize, range(workers)))

    digest = hashlib.sha256(
        (package / viewer_assets.ARCHIVE_FILENAME).read_bytes()
    ).hexdigest()
    assert extraction_count == 1
    assert len(set(results)) == 1
    assert results[0].parent.name == digest
    assert (results[0] / "index.html").read_text() == "<main>viewer</main>"
    assert set(cache.iterdir()) == {
        results[0].parent,
        cache / f".{digest}.lock",
    }


def test_archive_rejects_path_traversal_before_writing(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    _write_archive(
        package,
        {
            "static/index.html": b"<main>viewer</main>",
            "static/assets/app.js": b"app",
            "static/../../escaped": b"unsafe",
        },
    )
    destination = tmp_path / "destination"

    with pytest.raises(PrebuiltViewerAssetsError, match="unsafe"):
        extract_prebuilt_viewer_archive(
            package / viewer_assets.ARCHIVE_FILENAME,
            package / viewer_assets.DIGEST_FILENAME,
            destination,
        )

    assert not (tmp_path / "escaped").exists()
    assert not (destination / "static").exists()


def test_archive_digest_is_verified_before_extraction(tmp_path: Path) -> None:
    package = tmp_path / "package"
    _write_archive(
        package,
        {
            "static/index.html": b"<main>viewer</main>",
            "static/assets/app.js": b"app",
        },
    )
    (package / viewer_assets.DIGEST_FILENAME).write_text(
        "0" * 64 + "\n", encoding="ascii"
    )

    with pytest.raises(PrebuiltViewerAssetsError, match="digest"):
        extract_prebuilt_viewer_archive(
            package / viewer_assets.ARCHIVE_FILENAME,
            package / viewer_assets.DIGEST_FILENAME,
            tmp_path / "destination",
        )

    assert not (tmp_path / "destination").exists()


def _write_archive(package: Path, files: dict[str, bytes]) -> None:
    package.mkdir(parents=True)
    stream = io.BytesIO()
    with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, content in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
    archive_bytes = stream.getvalue()
    (package / viewer_assets.ARCHIVE_FILENAME).write_bytes(archive_bytes)
    (package / viewer_assets.DIGEST_FILENAME).write_text(
        hashlib.sha256(archive_bytes).hexdigest() + "\n",
        encoding="ascii",
    )
