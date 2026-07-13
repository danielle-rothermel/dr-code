from __future__ import annotations

import subprocess

import pytest

from dr_code.humaneval import sandbox


def test_runtime_environment_excludes_operator_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("DATABASE_URL", "secret")
    monkeypatch.setenv("PATH", "/bin")

    environment = sandbox._runtime_environment()
    assert environment["PATH"] == "/bin"
    assert "ANTHROPIC_API_KEY" not in environment
    assert "DATABASE_URL" not in environment


@pytest.mark.parametrize(
    "image",
    [
        "python:3.13-slim",
        "python:3.13-slim@sha256:short",
        "python:3.13-slim@sha256:" + "z" * 64,
        "python:3.13 slim@sha256:" + "a" * 64,
        "@sha256:" + "a" * 64,
    ],
)
def test_mutable_or_malformed_image_reference_is_rejected(image: str) -> None:
    with pytest.raises(sandbox.SandboxError, match="immutable sha256"):
        sandbox._validate_image_reference(image)


def test_hung_image_inspection_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def hang(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="docker", timeout=10)

    monkeypatch.setattr(sandbox.subprocess, "run", hang)

    with pytest.raises(sandbox.SandboxError, match="image inspection failed"):
        sandbox._require_local_image("docker", sandbox.SANDBOX_IMAGE)


def test_missing_runtime_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DR_CODE_SANDBOX_RUNTIME", "docker")
    monkeypatch.setattr(sandbox.shutil, "which", lambda _: None)

    with pytest.raises(sandbox.SandboxError, match="runtime is unavailable"):
        sandbox._resolve_runtime()
