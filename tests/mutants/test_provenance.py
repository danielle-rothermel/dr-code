"""Content-bound production runner provenance contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from dr_code.mutants import provenance as provenance_module


def test_production_runner_identity_payload_and_hash_are_golden() -> None:
    source_bytes = b"pass\n"
    package_digest = "a" * 64

    payload = provenance_module._production_runner_identity_payload(
        runner_source_utf8=source_bytes,
        dr_code_python_package_sha256=package_digest,
    )

    assert payload == {
        "dr_code_python_package_sha256": package_digest,
        "python_argv_prefix": ["-I", "-c"],
        "runner_source_utf8_sha256": (
            "9f56e761d79bfdb34304a012586cb04d16b435ef6130091a97702e559260a2f2"
        ),
        "runner_source_utf8_size": 5,
    }
    assert provenance_module._production_runner_identity(payload) == (
        provenance_module.PRODUCTION_RUNNER_IDENTITY_PREFIX
        + "28fde655e238694392f348891164aa1f3b2aee9620a945b9a4ac4219"
        "e3a3255c"
    )


def test_one_byte_runner_source_mutation_changes_identity() -> None:
    package_digest = "b" * 64
    original = provenance_module._production_runner_identity_payload(
        runner_source_utf8=b"pass\n",
        dr_code_python_package_sha256=package_digest,
    )
    changed = provenance_module._production_runner_identity_payload(
        runner_source_utf8=b"pass!\n",
        dr_code_python_package_sha256=package_digest,
    )

    assert provenance_module._production_runner_identity(
        original
    ) != provenance_module._production_runner_identity(changed)


def test_package_implementation_digest_changes_runner_identity() -> None:
    original = provenance_module._production_runner_identity_payload(
        runner_source_utf8=b"pass\n",
        dr_code_python_package_sha256="a" * 64,
    )
    changed = provenance_module._production_runner_identity_payload(
        runner_source_utf8=b"pass\n",
        dr_code_python_package_sha256=("a" * 63 + "b"),
    )

    assert provenance_module._production_runner_identity(
        original
    ) != provenance_module._production_runner_identity(changed)


def test_python_runtime_payload_and_identity_are_golden() -> None:
    coordinates = provenance_module._PythonRuntimeCoordinates(
        byteorder="little",
        implementation_cache_tag="cpython-313",
        implementation_hexversion=0x030D02F0,
        implementation_name="cpython",
        machine="arm64",
        python_executable_invoked_path="/venv/bin/python",
        python_executable_real_path="/runtime/bin/python3.13",
        python_executable_sha256="c" * 64,
        python_executable_size=49_968,
        python_version=(
            "3.13.2 (main, Feb 12 2025, 14:59:08) [Clang 19.1.6 ]"
        ),
        system="Darwin",
        system_release="24.6.0",
    )

    payload = coordinates.identity_payload()

    assert payload == {
        "byteorder": "little",
        "implementation_cache_tag": "cpython-313",
        "implementation_hexversion": 0x030D02F0,
        "implementation_name": "cpython",
        "machine": "arm64",
        "python_executable_invoked_path": "/venv/bin/python",
        "python_executable_real_path": "/runtime/bin/python3.13",
        "python_executable_sha256": "c" * 64,
        "python_executable_size": 49_968,
        "python_version": (
            "3.13.2 (main, Feb 12 2025, 14:59:08) [Clang 19.1.6 ]"
        ),
        "system": "Darwin",
        "system_release": "24.6.0",
    }
    assert provenance_module._runtime_identity(payload) == (
        "3abeb002c194f0a3929898a7e95771f9c67da5b8142391c5760d24c50d8e07b4"
    )


def test_capture_fails_closed_without_executable_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        provenance_module,
        "package_source_digest",
        lambda: "a" * 64,
    )
    monkeypatch.setattr(
        provenance_module.sys,
        "executable",
        str(tmp_path / "missing-python"),
    )

    with pytest.raises(
        provenance_module.RunnerProvenanceError,
        match="executable evidence is unavailable",
    ):
        provenance_module.capture_production_runner("pass\n")
