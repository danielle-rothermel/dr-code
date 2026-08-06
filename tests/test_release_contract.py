from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "check_release_contract.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "dr_code_release_contract_check", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
validate_release_contract = _MODULE.validate_release_contract


def _pyproject(version: str) -> bytes:
    return f'[project]\nname = "dr-code"\nversion = "{version}"\n'.encode()


def _lock(version: str) -> bytes:
    return (
        f'version = 1\n[[package]]\nname = "dr-code"\n'
        f'version = "{version}"\nsource = {{ editable = "." }}\n'
    ).encode()


def _changelog(version: str) -> str:
    return f"# Changelog\n\n## {version} - 2026-08-06\n\n- Change.\n"


def _validate(
    *,
    head_version: str = "1.2.4",
    lock_version: str = "1.2.4",
    changelog_version: str = "1.2.4",
) -> tuple[str, ...]:
    return validate_release_contract(
        base_pyproject=_pyproject("1.2.3"),
        head_pyproject=_pyproject(head_version),
        head_lock=_lock(lock_version),
        head_changelog=_changelog(changelog_version),
    )


def test_release_contract_accepts_one_patch_with_matching_metadata() -> None:
    assert _validate() == ()


@pytest.mark.parametrize(
    "violation",
    [
        lambda: _validate(head_version="1.2.3"),
        lambda: _validate(head_version="1.3.0"),
        lambda: _validate(lock_version="1.2.3"),
        lambda: _validate(changelog_version="1.2.3"),
    ],
)
def test_release_contract_rejects_each_inconsistent_version(
    violation: Callable[[], tuple[str, ...]],
) -> None:
    assert violation()
