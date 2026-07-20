"""Conscious exception: tests ``sandbox._*`` helpers directly.

These private helpers pin fail-closed security invariants (credential
scrubbing, immutable-digest enforcement, local-image inspection, runtime
allow-list) that must be verifiable without a live container runtime. The
same invariants are covered end-to-end by the CI-run OCI probes in
``tests/humaneval/test_sandbox.py``. The helpers must not be promoted to
public API to satisfy testing convention: the sandbox's minimal 3-kwarg
interface (``run_python_in_sandbox``) is deliberate.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest

from dr_code.humaneval import sandbox


def test_humaneval_plus_input_budget_includes_largest_payload_and_source() -> (
    None
):
    largest_humaneval_plus_payload_bytes = 1_255_579
    maximum_escaped_candidate_bytes = 512 * 1024

    assert sandbox.MAX_SANDBOX_INPUT_BYTES == 4 * 1024 * 1024
    assert (
        largest_humaneval_plus_payload_bytes + maximum_escaped_candidate_bytes
        <= sandbox.MAX_SANDBOX_INPUT_BYTES
    )


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


@pytest.mark.parametrize(
    "image",
    [
        "python:3.13-slim@sha256:" + "a" * 64,
        "sha256:" + "a" * 64,
    ],
)
def test_immutable_named_reference_or_local_image_id_is_accepted(
    image: str,
) -> None:
    sandbox._validate_image_reference(image)


def test_humaneval_plus_image_dockerfile_pins_runtime_and_numpy() -> None:
    repository_root = Path(__file__).parents[2]
    dockerfile = (
        repository_root / "docker" / "humaneval-plus" / "Dockerfile"
    ).read_text()

    assert f"FROM {sandbox.SANDBOX_IMAGE}" in dockerfile
    assert "numpy==2.2.6" in dockerfile


def test_sandbox_sets_single_openblas_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command: list[str] = []

    class CompletedProcess:
        returncode = 0
        stdin = io.BytesIO()
        stdout = io.BytesIO()
        stderr = io.BytesIO()

        def poll(self) -> int:
            return self.returncode

    def start(command_argument: list[str], **_: object) -> CompletedProcess:
        command.extend(command_argument)
        return CompletedProcess()

    monkeypatch.setattr(sandbox, "_resolve_runtime", lambda: "docker")
    monkeypatch.setattr(sandbox, "_require_local_image", lambda *_: None)
    monkeypatch.setattr(sandbox.subprocess, "Popen", start)

    sandbox.run_python_in_sandbox(
        source="input()\n",
        input_json="{}",
        timeout_seconds=1.0,
    )

    assert "--env=OPENBLAS_NUM_THREADS=1" in command


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


def test_unlisted_runtime_fails_closed_before_which_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DR_CODE_SANDBOX_RUNTIME", "podman-remote")

    def fail(_: str) -> None:
        raise AssertionError("which must not run for an unlisted runtime")

    monkeypatch.setattr(sandbox.shutil, "which", fail)

    with pytest.raises(sandbox.SandboxError, match="docker.*podman"):
        sandbox._resolve_runtime()
