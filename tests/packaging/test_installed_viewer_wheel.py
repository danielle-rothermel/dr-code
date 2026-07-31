from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from zipfile import ZipFile

import pytest

from dr_code.synthetic.humaneval_loader import (
    SNAPSHOT_RESOURCE,
    SNAPSHOT_SHA256,
)

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DATE_EPOCH = "1704067200"


@pytest.mark.xfail(
    reason=(
        "dr-exec is a local path source until it publishes; re-enable at the "
        "pin-swap commit"
    ),
)
def test_installed_wheel_serves_packaged_viewer(tmp_path: Path) -> None:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
    }
    environment["SOURCE_DATE_EPOCH"] = _SOURCE_DATE_EPOCH
    source_wheel_dir = tmp_path / "source-wheel"
    source_distribution_dir = tmp_path / "sdist"
    rebuilt_wheel_dir = tmp_path / "rebuilt-wheel"
    for path in (
        source_wheel_dir,
        source_distribution_dir,
        rebuilt_wheel_dir,
    ):
        path.mkdir()

    _run(
        "uv",
        "build",
        "--wheel",
        "--out-dir",
        str(source_wheel_dir),
        cwd=_ROOT,
        environment=environment,
    )
    _run(
        "uv",
        "build",
        "--sdist",
        "--out-dir",
        str(source_distribution_dir),
        cwd=_ROOT,
        environment=environment,
    )
    source_distribution = _one_artifact(source_distribution_dir, "*.tar.gz")
    _assert_sdist_contains_prebuilt_frontend(source_distribution)
    no_javascript_tools = tmp_path / "no-javascript-tools"
    no_javascript_tools.mkdir()
    for name in ("node", "pnpm", "corepack"):
        executable = no_javascript_tools / name
        executable.write_text(
            "#!/bin/sh\nexit 97\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
    wheel_environment = {
        **environment,
        "PATH": (
            f"{no_javascript_tools}{os.pathsep}{environment.get('PATH', '')}"
        ),
    }
    _run(
        "uv",
        "build",
        "--wheel",
        "--out-dir",
        str(rebuilt_wheel_dir),
        str(source_distribution),
        cwd=_ROOT,
        environment=wheel_environment,
    )

    source_wheel = _one_artifact(source_wheel_dir, "*.whl")
    rebuilt_wheel = _one_artifact(rebuilt_wheel_dir, "*.whl")
    assert _sha256(source_wheel) == _sha256(rebuilt_wheel)
    _assert_wheel_contains_only_packaged_frontend(source_wheel)

    project = tmp_path / "installed"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        "\n".join(
            (
                "[project]",
                'name = "installed-viewer-smoke"',
                'version = "0.0.0"',
                'requires-python = ">=3.13"',
                "dependencies = [",
                f'  "dr-code @ {source_wheel.as_uri()}",',
                '  "httpx>=0.28.1",',
                "]",
                "",
            )
        ),
        encoding="utf-8",
    )
    _run(
        "uv",
        "sync",
        "--project",
        str(project),
        "--no-dev",
        "--python",
        sys.executable,
        cwd=project,
        environment=environment,
    )
    help_text = _run(
        "uv",
        "run",
        "--project",
        str(project),
        "--no-sync",
        "dr-code",
        "--help",
        cwd=project,
        environment=environment,
    )
    assert "viewer" in help_text
    result = _run(
        "uv",
        "run",
        "--project",
        str(project),
        "--no-sync",
        "python",
        "-c",
        _INSTALLED_HTTP_SMOKE,
        cwd=project,
        environment=environment,
    )
    assert result.strip() == "installed viewer smoke passed"


def _assert_sdist_contains_prebuilt_frontend(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        _assert_root_readme_markdown_targets_are_packaged(names)
        asset_archive_name = next(
            name
            for name in names
            if name.endswith(
                "/src/dr_code/viewer/prebuilt_viewer_assets.tar.gz"
            )
        )
        digest_name = next(
            name
            for name in names
            if name.endswith(
                "/src/dr_code/viewer/prebuilt_viewer_assets.sha256"
            )
        )
        asset_stream = archive.extractfile(asset_archive_name)
        digest_stream = archive.extractfile(digest_name)
        assert asset_stream is not None
        assert digest_stream is not None
        asset_bytes = asset_stream.read()
        recorded_digest = digest_stream.read().decode("ascii").strip()
    assert hashlib.sha256(asset_bytes).hexdigest() == recorded_digest
    with tarfile.open(fileobj=io.BytesIO(asset_bytes), mode="r:gz") as assets:
        asset_names = assets.getnames()
    assert "static/index.html" in asset_names
    assert any(name.startswith("static/assets/") for name in asset_names)
    assert any(name.endswith("/hatch_build.py") for name in names)
    assert not any("/src/dr_code/viewer/static/" in name for name in names)
    assert not any("/node_modules/" in name for name in names)
    assert not any(
        "/viewer/packages/" in name and "/dist/" in name for name in names
    )


def _assert_root_readme_markdown_targets_are_packaged(
    archive_names: list[str],
) -> None:
    targets = {
        match.split("#", 1)[0]
        for match in re.findall(
            r"\[[^\]]+\]\(([^)]+\.md(?:#[^)]*)?)\)",
            (_ROOT / "README.md").read_text(encoding="utf-8"),
        )
        if "://" not in match
    }
    assert targets
    for target in targets:
        assert (_ROOT / target).is_file()
        assert any(name.endswith(f"/{target}") for name in archive_names)


def _assert_wheel_contains_only_packaged_frontend(path: Path) -> None:
    with ZipFile(path) as archive:
        names = archive.namelist()
        snapshot = archive.read(f"dr_code/synthetic/{SNAPSHOT_RESOURCE}")
    assert "dr_code/viewer/static/index.html" in names
    assert any(
        name.startswith("dr_code/viewer/static/assets/")
        and not name.endswith("/")
        for name in names
    )
    assert not any(
        "viewer/packages/preprocessing-analysis/dist" in name for name in names
    )
    assert not any("prebuilt_viewer_assets" in name for name in names)
    assert not any(
        name.endswith(".parquet") or "/analysis/" in name for name in names
    )
    assert hashlib.sha256(snapshot).hexdigest() == SNAPSHOT_SHA256


def _one_artifact(directory: Path, pattern: str) -> Path:
    artifacts = tuple(directory.glob(pattern))
    assert len(artifacts) == 1
    return artifacts[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(
    *command: str,
    cwd: Path,
    environment: dict[str, str],
) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"command failed: {json.dumps(command)}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return completed.stdout


_INSTALLED_HTTP_SMOKE = """
import re

from fastapi.testclient import TestClient

from dr_code.viewer.app import create_app
from dr_code.synthetic.humaneval_loader import load_humaneval_plus


class Service:
    def list_runs(self):
        return ()


client = TestClient(create_app(Service()), base_url="http://127.0.0.1")
root = client.get("/")
assert root.status_code == 200
assert 'id="root"' in root.text
fallback = client.get("/review/installed-wheel")
assert fallback.status_code == 200
assert fallback.content == root.content
assets = re.findall(r'(?:src|href)="(/assets/[^"]+)"', root.text)
assert assets
asset = client.get(assets[0])
assert asset.status_code == 200
assert asset.content != root.content
api = client.get("/api/runs")
assert api.status_code == 200
assert api.json() == []
tasks = load_humaneval_plus()
assert tasks[0].task_id == "HumanEval/0"
print("installed viewer smoke passed")
"""
