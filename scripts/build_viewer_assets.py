"""Build or verify the immutable preprocessing-viewer asset archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
PNPM_VERSION = "11.9.0"
NODE_MAJOR_VERSION = 22
_APPLICATION_PACKAGE = "@dr-code/preprocessing-analysis"
_IGNORED_BUILD_DIRECTORIES = frozenset(
    {
        ".vite",
        "coverage",
        "dist",
        "node_modules",
        "playwright-report",
        "test-results",
    }
)


@dataclass(frozen=True, slots=True)
class _AssetArtifacts:
    archive: bytes
    digest: bytes


class _CommandRunner(Protocol):
    def __call__(
        self,
        *command: str,
        cwd: Path,
        expected: str | None = None,
        expected_major: int | None = None,
    ) -> str: ...


def main(argv: Sequence[str] | None = None) -> None:
    """Build the frontend and either verify or replace checked artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if checked assets differ without changing the worktree",
    )
    arguments = parser.parse_args(argv)
    build_viewer_assets(ROOT, check=arguments.check, run_command=_run)


def build_viewer_assets(
    root: Path,
    *,
    check: bool,
    run_command: _CommandRunner,
) -> None:
    """Build canonical artifacts and check or replace each file atomically."""
    artifacts = _build_artifacts(root, run_command=run_command)
    archive_path, digest_path = _artifact_paths(root)
    if check:
        _check_artifacts(archive_path, digest_path, artifacts)
        return
    _write_artifacts(archive_path, digest_path, artifacts)


def _build_artifacts(
    root: Path, *, run_command: _CommandRunner
) -> _AssetArtifacts:
    workspace = root / "viewer"
    _validate_declared_toolchain(workspace)
    run_command(
        "node",
        "--version",
        cwd=root,
        expected_major=NODE_MAJOR_VERSION,
    )
    run_command(
        "corepack",
        f"pnpm@{PNPM_VERSION}",
        "--version",
        cwd=root,
        expected=PNPM_VERSION,
    )
    with TemporaryDirectory(prefix="dr-code-viewer-build-") as temporary:
        temporary_root = Path(temporary)
        temporary_workspace = temporary_root / "viewer"
        shutil.copytree(
            workspace,
            temporary_workspace,
            ignore=_ignore_generated_directories,
        )
        static_directory = (
            temporary_root / "src" / "dr_code" / "viewer" / "static"
        )
        run_command(
            "corepack",
            f"pnpm@{PNPM_VERSION}",
            "install",
            "--frozen-lockfile",
            cwd=temporary_workspace,
        )
        run_command(
            "corepack",
            f"pnpm@{PNPM_VERSION}",
            "--filter",
            _APPLICATION_PACKAGE,
            "exec",
            "tsc",
            "--noEmit",
            "--pretty",
            "false",
            cwd=temporary_workspace,
        )
        run_command(
            "corepack",
            f"pnpm@{PNPM_VERSION}",
            "--filter",
            _APPLICATION_PACKAGE,
            "exec",
            "vite",
            "build",
            "--outDir",
            str(static_directory),
            "--emptyOutDir",
            cwd=temporary_workspace,
        )
        _validate_static_directory(static_directory)
        return _create_artifacts(static_directory)


def _artifact_paths(root: Path) -> tuple[Path, Path]:
    package = root / "src" / "dr_code" / "viewer"
    return (
        package / "prebuilt_viewer_assets.tar.gz",
        package / "prebuilt_viewer_assets.sha256",
    )


def _validate_declared_toolchain(workspace: Path) -> None:
    package = json.loads(
        (workspace / "package.json").read_text(encoding="utf-8")
    )
    if package.get("packageManager") != f"pnpm@{PNPM_VERSION}":
        raise RuntimeError(
            "viewer packageManager does not match the asset builder"
        )
    engines = package.get("engines")
    if not isinstance(engines, dict) or engines.get("node") != ">=22 <23":
        raise RuntimeError(
            "viewer Node engine does not match the asset builder"
        )


def _ignore_generated_directories(
    directory: str, names: list[str]
) -> list[str]:
    parent = Path(directory)
    return [
        name
        for name in names
        if name in _IGNORED_BUILD_DIRECTORIES and (parent / name).is_dir()
    ]


def _run(
    *command: str,
    cwd: Path,
    expected: str | None = None,
    expected_major: int | None = None,
) -> str:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("VITE_")
    }
    environment.update(
        {
            "CI": "true",
            "LC_ALL": "C",
            "SOURCE_DATE_EPOCH": "0",
            "TZ": "UTC",
        }
    )
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"asset command failed: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    output = completed.stdout.strip()
    if expected is not None and output != expected:
        raise RuntimeError(
            f"expected {' '.join(command)} to report {expected}, got {output}"
        )
    if expected_major is not None:
        match = re.fullmatch(r"v(\d+)\.\d+\.\d+", output)
        if match is None or int(match.group(1)) != expected_major:
            raise RuntimeError(
                f"expected Node {expected_major}.x, got {output}"
            )
    return output


def _validate_static_directory(static_directory: Path) -> None:
    if not (static_directory / "index.html").is_file():
        raise RuntimeError("viewer build produced no index.html")
    assets = static_directory / "assets"
    if not assets.is_dir() or not any(
        path.is_file() for path in assets.iterdir()
    ):
        raise RuntimeError("viewer build produced no assets")
    if any(path.is_symlink() for path in static_directory.rglob("*")):
        raise RuntimeError("viewer build produced a symlink")


def _create_artifacts(static_directory: Path) -> _AssetArtifacts:
    stream = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=stream,
        mtime=0,
    ) as compressed:
        with tarfile.open(
            mode="w",
            fileobj=compressed,
            format=tarfile.PAX_FORMAT,
        ) as archive:
            _add_directory(archive, "static")
            for path in sorted(
                static_directory.rglob("*"),
                key=lambda item: item.relative_to(static_directory).as_posix(),
            ):
                relative = path.relative_to(static_directory)
                archive_name = (Path("static") / relative).as_posix()
                if path.is_dir():
                    _add_directory(archive, archive_name)
                elif path.is_file():
                    _add_file(archive, path, archive_name)
    archive_bytes = stream.getvalue()
    digest_bytes = (hashlib.sha256(archive_bytes).hexdigest() + "\n").encode(
        "ascii"
    )
    return _AssetArtifacts(archive=archive_bytes, digest=digest_bytes)


def _check_artifacts(
    archive_path: Path,
    digest_path: Path,
    expected: _AssetArtifacts,
) -> None:
    stale = []
    if (
        not archive_path.is_file()
        or archive_path.read_bytes() != expected.archive
    ):
        stale.append(archive_path.name)
    if (
        not digest_path.is_file()
        or digest_path.read_bytes() != expected.digest
    ):
        stale.append(digest_path.name)
    if stale:
        joined = ", ".join(stale)
        raise RuntimeError(
            f"checked viewer assets are stale ({joined}); "
            "run scripts/build_viewer_assets.py with Node 22"
        )


def _write_artifacts(
    archive_path: Path,
    digest_path: Path,
    artifacts: _AssetArtifacts,
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    staged_archive = _stage_artifact(archive_path, artifacts.archive)
    staged_digest = _stage_artifact(digest_path, artifacts.digest)
    try:
        os.replace(staged_archive, archive_path)
        os.replace(staged_digest, digest_path)
    finally:
        staged_archive.unlink(missing_ok=True)
        staged_digest.unlink(missing_ok=True)
    archive_path.chmod(0o644)
    digest_path.chmod(0o644)


def _stage_artifact(destination: Path, content: bytes) -> Path:
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(content)
        temporary.flush()
        os.fsync(temporary.fileno())
        return Path(temporary.name)


def _tar_info(name: str, *, mode: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.mode = mode
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def _add_directory(archive: tarfile.TarFile, name: str) -> None:
    info = _tar_info(name.rstrip("/") + "/", mode=0o755)
    info.type = tarfile.DIRTYPE
    archive.addfile(info)


def _add_file(archive: tarfile.TarFile, path: Path, name: str) -> None:
    info = _tar_info(name, mode=0o644)
    info.size = path.stat().st_size
    with path.open("rb") as stream:
        archive.addfile(info, stream)


if __name__ == "__main__":
    main()
