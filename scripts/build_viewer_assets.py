"""Build the one immutable preprocessing-viewer asset archive."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "viewer"
STATIC_DIRECTORY = ROOT / "src" / "dr_code" / "viewer" / "static"
ARCHIVE_PATH = (
    ROOT / "src" / "dr_code" / "viewer" / "prebuilt_viewer_assets.tar.gz"
)
DIGEST_PATH = (
    ROOT / "src" / "dr_code" / "viewer" / "prebuilt_viewer_assets.sha256"
)
PNPM_VERSION = "11.9.0"
NODE_MAJOR_VERSION = 22


def main() -> None:
    """Verify the pinned toolchain, build assets, and replace the archive."""
    _validate_declared_toolchain()
    _run("node", "--version", expected_major=NODE_MAJOR_VERSION)
    _run(
        "corepack",
        f"pnpm@{PNPM_VERSION}",
        "--version",
        expected=PNPM_VERSION,
    )
    _run(
        "corepack",
        f"pnpm@{PNPM_VERSION}",
        "install",
        "--frozen-lockfile",
        cwd=WORKSPACE,
    )
    _run(
        "corepack",
        f"pnpm@{PNPM_VERSION}",
        "--filter",
        "@dr-code/preprocessing-analysis",
        "build",
        cwd=WORKSPACE,
    )
    _validate_static_directory()
    _write_archive()


def _validate_declared_toolchain() -> None:
    package = json.loads(
        (WORKSPACE / "package.json").read_text(encoding="utf-8")
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


def _run(
    *command: str,
    cwd: Path = ROOT,
    expected: str | None = None,
    expected_major: int | None = None,
) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env={**os.environ, "CI": "true"},
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"asset command failed: {' '.join(command)}\n{completed.stderr}"
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


def _validate_static_directory() -> None:
    if not (STATIC_DIRECTORY / "index.html").is_file():
        raise RuntimeError("viewer build produced no index.html")
    assets = STATIC_DIRECTORY / "assets"
    if not assets.is_dir() or not any(
        path.is_file() for path in assets.iterdir()
    ):
        raise RuntimeError("viewer build produced no assets")
    if any(path.is_symlink() for path in STATIC_DIRECTORY.rglob("*")):
        raise RuntimeError("viewer build produced a symlink")


def _write_archive() -> None:
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=ARCHIVE_PATH.parent,
        prefix=f".{ARCHIVE_PATH.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=temporary,
            mtime=0,
        ) as compressed:
            with tarfile.open(
                mode="w",
                fileobj=compressed,
                format=tarfile.PAX_FORMAT,
            ) as archive:
                _add_directory(archive, "static")
                for path in sorted(
                    STATIC_DIRECTORY.rglob("*"),
                    key=lambda item: item.relative_to(
                        STATIC_DIRECTORY
                    ).as_posix(),
                ):
                    relative = path.relative_to(STATIC_DIRECTORY)
                    archive_name = (Path("static") / relative).as_posix()
                    if path.is_dir():
                        _add_directory(archive, archive_name)
                    elif path.is_file():
                        _add_file(archive, path, archive_name)
    try:
        os.replace(temporary_path, ARCHIVE_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)
    ARCHIVE_PATH.chmod(0o644)
    digest = hashlib.sha256(ARCHIVE_PATH.read_bytes()).hexdigest()
    DIGEST_PATH.write_text(digest + "\n", encoding="ascii")


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
