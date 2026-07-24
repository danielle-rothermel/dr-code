"""Canonical implementation evidence for installed dr-code Python sources."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast

from dr_serialize import (
    Jsonable,
    build_identity_document,
    identity_document_hash,
)

_PACKAGE_ROOT = Path(__file__).resolve().parent
_PACKAGE_NAME = "dr_code"


@dataclass(frozen=True, slots=True)
class PythonSourceManifestEntry:
    """One canonical path/content entry in the installed source manifest."""

    relative_path: str
    sha256: str

    def identity_payload(self) -> dict[str, str]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }


def package_source_manifest(
    package_root: Path = _PACKAGE_ROOT,
) -> tuple[PythonSourceManifestEntry, ...]:
    """Read every installed Python source file into a canonical manifest."""

    try:
        source_paths = sorted(
            path for path in package_root.rglob("*.py") if path.is_file()
        )
    except OSError as exc:
        raise ValueError(
            f"cannot enumerate Python package source evidence: {package_root}"
        ) from exc
    if not source_paths:
        raise ValueError(
            f"Python package source evidence is empty: {package_root}"
        )

    entries: list[PythonSourceManifestEntry] = []
    for source_path in source_paths:
        try:
            source_bytes = source_path.read_bytes()
            relative_path = source_path.relative_to(package_root).as_posix()
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"cannot read Python package source evidence: {source_path}"
            ) from exc
        entries.append(
            PythonSourceManifestEntry(
                relative_path=relative_path,
                sha256=hashlib.sha256(source_bytes).hexdigest(),
            )
        )
    return tuple(entries)


def _manifest_digest(
    manifest: tuple[PythonSourceManifestEntry, ...],
) -> str:
    document = build_identity_document(
        schema="dr_code.python_package_source_manifest",
        schema_version=1,
        payload=cast(
            Jsonable,
            [entry.identity_payload() for entry in manifest],
        ),
    )
    return identity_document_hash(document)


def package_source_digest(package_root: Path = _PACKAGE_ROOT) -> str:
    """Hash the complete canonical installed Python-package source manifest."""

    return _manifest_digest(package_source_manifest(package_root))


def implementation_identity_for(
    *,
    module: str,
    qualname: str,
    package_root: Path = _PACKAGE_ROOT,
) -> str:
    """Bind class coordinates to the complete package source artifact."""

    return _implementation_identity_from_manifest(
        module=module,
        qualname=qualname,
        manifest=package_source_manifest(package_root),
    )


def _implementation_identity_from_manifest(
    *,
    module: str,
    qualname: str,
    manifest: tuple[PythonSourceManifestEntry, ...],
) -> str:
    if not module or not qualname:
        raise ValueError(
            "implementation module and qualname must be non-empty"
        )
    module_parts = module.split(".")
    if not module_parts or module_parts[0] != _PACKAGE_NAME:
        raise ValueError(
            "implementation module must belong to the dr_code package"
        )

    module_relative = Path(*module_parts[1:])
    candidates = {
        module_relative.with_suffix(".py").as_posix(),
        (module_relative / "__init__.py").as_posix(),
    }
    if not candidates & {entry.relative_path for entry in manifest}:
        raise ValueError(
            f"implementation module has no package source evidence: {module}"
        )
    package_digest = _manifest_digest(manifest)
    document = build_identity_document(
        schema="dr_code.python_implementation",
        schema_version=1,
        payload=cast(
            Jsonable,
            {
                "module": module,
                "qualname": qualname,
                "package_artifact_sha256": package_digest,
            },
        ),
    )
    return identity_document_hash(document)


@lru_cache(maxsize=1)
def _installed_package_manifest() -> tuple[PythonSourceManifestEntry, ...]:
    return package_source_manifest(_PACKAGE_ROOT)


def implementation_identity(implementation: type[object]) -> str:
    """Return package-wide executable evidence for one implementation class."""

    return _implementation_identity_from_manifest(
        module=implementation.__module__,
        qualname=implementation.__qualname__,
        manifest=_installed_package_manifest(),
    )


__all__ = [
    "PythonSourceManifestEntry",
    "implementation_identity",
    "implementation_identity_for",
    "package_source_digest",
    "package_source_manifest",
]
