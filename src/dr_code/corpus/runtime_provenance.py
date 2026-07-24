"""Canonical installed-environment provenance shared by corpus producers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from typing import Final

_NORMALIZED_DISTRIBUTION_NAME: Final = re.compile(r"[-_.]+")


class RuntimeProvenanceError(ValueError):
    """The installed dependency environment cannot be represented exactly."""


def installed_environment_coordinates() -> list[dict[str, str]]:
    """Return sorted, normalized installed distribution name/version pairs."""

    coordinates: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        version = distribution.version
        if not name or not version:
            raise RuntimeProvenanceError(
                "installed distribution is missing name or version"
            )
        normalized = _NORMALIZED_DISTRIBUTION_NAME.sub("-", name).lower()
        existing = coordinates.get(normalized)
        if existing is not None and existing != version:
            raise RuntimeProvenanceError(
                "installed distribution has conflicting versions: "
                f"{normalized!r}"
            )
        coordinates[normalized] = version
    return [
        {"name": name, "version": version}
        for name, version in sorted(coordinates.items())
    ]


def installed_environment_provenance() -> dict[str, object]:
    """Return coordinates plus their canonical content identity."""

    coordinates = installed_environment_coordinates()
    return {
        "distributions": coordinates,
        "identity": hashlib.sha256(
            json.dumps(
                coordinates,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }


__all__ = [
    "RuntimeProvenanceError",
    "installed_environment_coordinates",
    "installed_environment_provenance",
]
