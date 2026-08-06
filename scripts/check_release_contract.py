#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).parents[1]
_VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


@dataclass(frozen=True, slots=True)
class _Version:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: object, *, source: str) -> _Version:
        if not isinstance(value, str):
            raise ValueError(f"{source} project version must be a string")
        match = _VERSION_PATTERN.fullmatch(value)
        if match is None:
            raise ValueError(
                f"{source} project version must have the form X.Y.Z"
            )
        return cls(*(int(part) for part in match.groups()))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def _toml_document(content: bytes, *, source: str) -> dict[str, object]:
    try:
        return tomllib.loads(content.decode())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"{source} is not valid UTF-8 TOML") from exc


def _project_version(content: bytes, *, source: str) -> _Version:
    document = _toml_document(content, source=source)
    project = document.get("project")
    if not isinstance(project, dict):
        raise ValueError(f"{source} has no project table")
    return _Version.parse(project.get("version"), source=source)


def _locked_project_version(content: bytes) -> _Version:
    document = _toml_document(content, source="uv.lock")
    packages = document.get("package")
    if not isinstance(packages, list):
        raise ValueError("uv.lock has no package entries")
    matches = [
        package
        for package in packages
        if isinstance(package, dict) and package.get("name") == "dr-code"
    ]
    if len(matches) != 1:
        raise ValueError("uv.lock must contain exactly one dr-code package")
    return _Version.parse(matches[0].get("version"), source="uv.lock")


def _has_dated_changelog_entry(content: str, version: _Version) -> bool:
    heading = re.compile(
        rf"^## {re.escape(str(version))} - (?P<date>\d{{4}}-\d{{2}}-\d{{2}})$",
        re.MULTILINE,
    )
    for match in heading.finditer(content):
        try:
            date.fromisoformat(match.group("date"))
        except ValueError:
            continue
        return True
    return False


def validate_release_contract(
    *,
    base_pyproject: bytes,
    head_pyproject: bytes,
    head_lock: bytes,
    head_changelog: str,
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        base = _project_version(base_pyproject, source="base pyproject.toml")
        head = _project_version(head_pyproject, source="head pyproject.toml")
    except ValueError as exc:
        return (str(exc),)

    expected = _Version(base.major, base.minor, base.patch + 1)
    if head != expected:
        errors.append(
            f"project version must advance exactly one patch: "
            f"expected {expected}, found {head}"
        )

    try:
        locked = _locked_project_version(head_lock)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if locked != head:
            errors.append(
                f"uv.lock dr-code version must be {head}, found {locked}"
            )

    if not _has_dated_changelog_entry(head_changelog, head):
        errors.append(
            f"CHANGELOG.md must contain a dated '## {head} - YYYY-MM-DD' "
            "heading"
        )
    return tuple(errors)


def _git_file(ref: str, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            cwd=_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except subprocess.CalledProcessError as exc:
        diagnostic = exc.stderr.decode(errors="replace").strip()
        raise ValueError(
            f"cannot read {path} from base ref {ref!r}: {diagnostic}"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the pull request release-version contract."
    )
    parser.add_argument(
        "--base-ref",
        required=True,
        help="exact base commit or local ref used for comparison",
    )
    arguments = parser.parse_args()

    try:
        base_pyproject = _git_file(arguments.base_ref, "pyproject.toml")
    except ValueError as exc:
        parser.error(str(exc))

    errors = validate_release_contract(
        base_pyproject=base_pyproject,
        head_pyproject=(_ROOT / "pyproject.toml").read_bytes(),
        head_lock=(_ROOT / "uv.lock").read_bytes(),
        head_changelog=(_ROOT / "CHANGELOG.md").read_text(),
    )
    if errors:
        for error in errors:
            print(f"release contract violation: {error}", file=sys.stderr)
        return 1

    print("release contract satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
