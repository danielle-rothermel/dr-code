"""Installed-package implementation evidence contracts."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from dr_code.implementation_identity import (
    implementation_identity_for,
    package_source_digest,
    package_source_manifest,
)


@pytest.fixture(scope="module")
def built_wheel_package(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    build_root = tmp_path_factory.mktemp("installed-wheel")
    wheel_dir = build_root / "dist"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    installed_root = build_root / "installed"
    with zipfile.ZipFile(wheels[0]) as archive:
        archive.extractall(installed_root)
    return installed_root / "dr_code"


def _write_package(root: Path) -> Path:
    package_root = root / "dr_code"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("")
    (package_root / "base.py").write_text(
        "class Base:\n    def execute(self):\n        return helper()\n"
    )
    (package_root / "helper.py").write_text("def helper():\n    return 1\n")
    (package_root / "operator.py").write_text(
        "from .base import Base\n"
        "class Implementation(Base):\n"
        "    VERSION = '1'\n"
    )
    return package_root


def test_package_manifest_is_location_stable(tmp_path: Path) -> None:
    source_root = _write_package(tmp_path / "source")
    installed_root = _write_package(tmp_path / "installed")

    assert package_source_manifest(source_root) == package_source_manifest(
        installed_root
    )
    assert package_source_digest(source_root) == package_source_digest(
        installed_root
    )
    assert implementation_identity_for(
        module="dr_code.operator",
        qualname="Implementation",
        package_root=source_root,
    ) == implementation_identity_for(
        module="dr_code.operator",
        qualname="Implementation",
        package_root=installed_root,
    )


def test_built_wheel_matches_source_package_manifest(
    built_wheel_package: Path,
) -> None:
    source_package = Path(__file__).resolve().parents[2] / "src" / "dr_code"

    assert package_source_manifest(
        built_wheel_package
    ) == package_source_manifest(source_package)
    assert package_source_digest(built_wheel_package) == package_source_digest(
        source_package
    )


def test_built_wheel_manifest_detects_mutated_installed_source(
    tmp_path: Path,
    built_wheel_package: Path,
) -> None:
    mutated_package = tmp_path / "dr_code"
    shutil.copytree(built_wheel_package, mutated_package)
    source = mutated_package / "corpus" / "candidate_evaluation.py"
    source.write_bytes(source.read_bytes() + b"\n# mutated installed bytes\n")

    assert package_source_manifest(mutated_package) != package_source_manifest(
        built_wheel_package
    )
    assert package_source_digest(mutated_package) != package_source_digest(
        built_wheel_package
    )


@pytest.mark.parametrize("dependency_path", ["helper.py", "base.py"])
def test_imported_dependency_bytes_change_implementation_identity(
    tmp_path: Path,
    dependency_path: str,
) -> None:
    original_root = _write_package(tmp_path / "original")
    changed_root = _write_package(tmp_path / "changed")
    dependency = changed_root / dependency_path
    dependency.write_text(dependency.read_text() + "# changed behavior\n")

    assert implementation_identity_for(
        module="dr_code.operator",
        qualname="Implementation",
        package_root=original_root,
    ) != implementation_identity_for(
        module="dr_code.operator",
        qualname="Implementation",
        package_root=changed_root,
    )


def test_package_manifest_fails_closed_without_source_evidence(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="source evidence is empty"):
        package_source_manifest(tmp_path)


def test_package_manifest_fails_closed_on_unreadable_source(
    tmp_path: Path,
) -> None:
    package_root = _write_package(tmp_path / "unreadable")
    source_path = package_root / "helper.py"
    source_path.chmod(0)
    try:
        with pytest.raises(ValueError, match="cannot read"):
            package_source_manifest(package_root)
    finally:
        source_path.chmod(0o600)
