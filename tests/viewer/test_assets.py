from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import sys
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import pytest

import dr_code.viewer.assets as viewer_assets
from dr_code.viewer.assets import (
    PrebuiltViewerAssetsError,
    extract_prebuilt_viewer_archive,
    materialize_prebuilt_viewer_assets,
)


def _load_asset_builder() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "build_viewer_assets.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_dr_code_test_asset_builder", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


asset_builder = _load_asset_builder()


def test_asset_check_rejects_archive_built_from_stale_source(
    tmp_path: Path,
) -> None:
    root = _write_asset_builder_fixture(tmp_path, source=b"first source")
    asset_builder.build_viewer_assets(
        root,
        check=False,
        run_command=_fake_asset_command,
    )
    source = root / "viewer" / "asset-source.txt"
    source.write_bytes(b"changed source")
    before_check = _file_snapshot(root)

    with pytest.raises(RuntimeError, match="viewer assets are stale"):
        asset_builder.build_viewer_assets(
            root,
            check=True,
            run_command=_fake_asset_command,
        )

    assert _file_snapshot(root) == before_check


def test_asset_build_is_deterministic_and_check_does_not_write(
    tmp_path: Path,
) -> None:
    root = _write_asset_builder_fixture(tmp_path, source=b"stable source")
    archive_path, digest_path = asset_builder._artifact_paths(root)

    asset_builder.build_viewer_assets(
        root,
        check=False,
        run_command=_fake_asset_command,
    )
    first = (archive_path.read_bytes(), digest_path.read_bytes())
    asset_builder.build_viewer_assets(
        root,
        check=False,
        run_command=_fake_asset_command,
    )
    second = (archive_path.read_bytes(), digest_path.read_bytes())

    assert second == first
    before_check = _file_snapshot(root)
    asset_builder.build_viewer_assets(
        root,
        check=True,
        run_command=_fake_asset_command,
    )
    assert _file_snapshot(root) == before_check


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

    with ThreadPoolExecutor(max_workers=workers) as executor:
        warm_results = list(executor.map(materialize, range(workers)))

    assert warm_results == results
    assert extraction_count == 1


@pytest.mark.parametrize("tamper", ["modified", "extra"])
def test_materialization_repairs_tampered_persistent_cache(
    tmp_path: Path, tamper: str
) -> None:
    package = tmp_path / "package"
    canonical = b"console.log('viewer')"
    _write_archive(
        package,
        {
            "static/index.html": b"<main>viewer</main>",
            "static/assets/app.js": canonical,
        },
    )
    cache = tmp_path / "cache"
    static_dir = materialize_prebuilt_viewer_assets(package, cache_root=cache)
    if tamper == "modified":
        (static_dir / "assets" / "app.js").write_bytes(b"tampered")
    else:
        (static_dir / "assets" / "injected.js").write_bytes(b"extra")

    repaired = materialize_prebuilt_viewer_assets(package, cache_root=cache)

    assert repaired == static_dir
    assert (repaired / "assets" / "app.js").read_bytes() == canonical
    assert not (repaired / "assets" / "injected.js").exists()


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


def _write_asset_builder_fixture(tmp_path: Path, *, source: bytes) -> Path:
    root = tmp_path / "repository"
    workspace = root / "viewer"
    workspace.mkdir(parents=True)
    (workspace / "package.json").write_text(
        """{
  "packageManager": "pnpm@11.9.0",
  "engines": {"node": ">=22 <23"}
}
""",
        encoding="utf-8",
    )
    (workspace / "pnpm-lock.yaml").write_text(
        "lockfileVersion: '9.0'\n",
        encoding="utf-8",
    )
    (workspace / "asset-source.txt").write_bytes(source)
    return root


def _fake_asset_command(
    *command: str,
    cwd: Path,
    expected: str | None = None,
    expected_major: int | None = None,
) -> str:
    if command == ("node", "--version"):
        assert expected_major == 22
        return "v22.0.0"
    if command[-1] == "--version":
        assert expected == "11.9.0"
        return "11.9.0"
    if "install" in command:
        assert "--frozen-lockfile" in command
        return ""
    if "vite" in command:
        output = Path(command[command.index("--outDir") + 1])
        assets = output / "assets"
        assets.mkdir(parents=True)
        source = (cwd / "asset-source.txt").read_bytes()
        (output / "index.html").write_bytes(b"<main>viewer</main>")
        (assets / "app.js").write_bytes(source)
        return ""
    assert "tsc" in command
    return ""


def _file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
