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
import os
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
        cidfile_index = command_argument.index("--cidfile") + 1
        Path(command_argument[cidfile_index]).write_text("a" * 64)
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
    assert "--cidfile" in command


def test_runtime_exit_before_container_creation_fails_as_infrastructure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedProcess:
        returncode = 125
        stdin = io.BytesIO()
        stdout = io.BytesIO()
        stderr = io.BytesIO(b"daemon unavailable")

        def poll(self) -> int:
            return self.returncode

    cleaned: list[str] = []
    monkeypatch.setattr(sandbox, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(sandbox, "_resolve_runtime", lambda: "docker")
    monkeypatch.setattr(sandbox, "_require_local_image", lambda *_: None)
    monkeypatch.setattr(
        sandbox.subprocess, "Popen", lambda *_, **__: FailedProcess()
    )
    monkeypatch.setattr(
        sandbox,
        "_terminate_container",
        lambda _runtime, target, _process: cleaned.append(target),
    )

    with pytest.raises(
        sandbox.SandboxError, match="before producing a container ID"
    ):
        sandbox.run_python_in_sandbox(
            source="pass\n", input_json="{}", timeout_seconds=1.0
        )

    assert not list(tmp_path.glob("*.cid"))
    assert len(cleaned) == 1


def test_empty_cidfile_is_not_accepted_as_container_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedProcess:
        returncode = 125
        stdin = io.BytesIO()
        stdout = io.BytesIO()
        stderr = io.BytesIO(b"cidfile write failed")

        def poll(self) -> int:
            return self.returncode

    def start(command: list[str], **_: object) -> FailedProcess:
        cidfile_index = command.index("--cidfile") + 1
        Path(command[cidfile_index]).touch()
        return FailedProcess()

    cleaned: list[str] = []
    monkeypatch.setattr(sandbox, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(sandbox, "_resolve_runtime", lambda: "docker")
    monkeypatch.setattr(sandbox, "_require_local_image", lambda *_: None)
    monkeypatch.setattr(sandbox.subprocess, "Popen", start)
    monkeypatch.setattr(
        sandbox,
        "_terminate_container",
        lambda _runtime, target, _process: cleaned.append(target),
    )

    with pytest.raises(
        sandbox.SandboxError, match="before producing a container ID"
    ):
        sandbox.run_python_in_sandbox(
            source="pass\n", input_json="{}", timeout_seconds=1.0
        )

    assert not list(tmp_path.glob("*.cid"))
    assert len(cleaned) == 1


def test_candidate_can_return_a_runtime_reserved_exit_code_after_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_id = "b" * 64

    class FailedProcess:
        returncode = 125
        stdin = io.BytesIO()
        stdout = io.BytesIO()
        stderr = io.BytesIO(b"daemon unavailable")

        def poll(self) -> int:
            return self.returncode

    def start(command: list[str], **_: object) -> FailedProcess:
        cidfile_index = command.index("--cidfile") + 1
        Path(command[cidfile_index]).write_text(container_id)
        return FailedProcess()

    monkeypatch.setattr(sandbox, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(sandbox, "_resolve_runtime", lambda: "docker")
    monkeypatch.setattr(sandbox, "_require_local_image", lambda *_: None)
    monkeypatch.setattr(sandbox.subprocess, "Popen", start)
    result = sandbox.run_python_in_sandbox(
        source="pass\n", input_json="{}", timeout_seconds=1.0
    )

    assert result.returncode == 125


def test_startup_interrupt_still_terminates_process_and_unlinks_cidfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RunningProcess:
        pid = 123
        returncode: int | None = None
        stdin = io.BytesIO()
        stdout = io.BytesIO()
        stderr = io.BytesIO()

        def poll(self) -> int | None:
            return self.returncode

    process = RunningProcess()
    terminated: list[str] = []

    def terminate(runtime: str, name: str, active_process: object) -> None:
        assert runtime == "docker"
        assert active_process is process
        terminated.append(name)
        process.returncode = -9

    monkeypatch.setattr(sandbox, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(sandbox, "_resolve_runtime", lambda: "docker")
    monkeypatch.setattr(sandbox, "_require_local_image", lambda *_: None)
    monkeypatch.setattr(sandbox.subprocess, "Popen", lambda *_, **__: process)
    monkeypatch.setattr(sandbox, "_terminate_container", terminate)
    monkeypatch.setattr(
        sandbox.time,
        "sleep",
        lambda _: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        sandbox.run_python_in_sandbox(
            source="input()\n", input_json="{}", timeout_seconds=1.0
        )

    assert len(terminated) == 1
    assert not list(tmp_path.glob("*.cid"))


def test_cleanup_failure_replaces_and_chains_a_startup_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RunningProcess:
        pid = 123
        returncode: int | None = None
        stdin = io.BytesIO()
        stdout = io.BytesIO()
        stderr = io.BytesIO()

        def poll(self) -> int | None:
            return self.returncode

    monkeypatch.setattr(sandbox, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(sandbox, "_resolve_runtime", lambda: "docker")
    monkeypatch.setattr(sandbox, "_require_local_image", lambda *_: None)
    monkeypatch.setattr(
        sandbox.subprocess, "Popen", lambda *_, **__: RunningProcess()
    )
    monkeypatch.setattr(
        sandbox,
        "_terminate_container",
        lambda *_: (_ for _ in ()).throw(
            sandbox.SandboxError("cleanup failed")
        ),
    )
    monkeypatch.setattr(
        sandbox.time,
        "sleep",
        lambda _: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(sandbox.SandboxError, match="cleanup failed") as error:
        sandbox.run_python_in_sandbox(
            source="input()\n", input_json="{}", timeout_seconds=1.0
        )

    assert isinstance(error.value.__cause__, KeyboardInterrupt)
    assert not list(tmp_path.glob("*.cid"))


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


def test_cleanup_falls_back_to_direct_child_kill_when_group_kill_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        pid = 123
        killed = False

        def poll(self) -> None:
            return None

        def kill(self) -> None:
            self.killed = True

        def wait(self, *, timeout: float) -> int:
            assert timeout == 5
            return -9

    process = Process()

    def deny_group_kill(pid: int, signal: int) -> None:
        assert pid == process.pid
        raise PermissionError(1, os.strerror(1))

    cleanup_results = iter(
        [
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 1),
            subprocess.CompletedProcess([], 0),
        ]
    )
    monkeypatch.setattr(sandbox.subprocess, "run", lambda *_, **__: None)
    monkeypatch.setattr(sandbox.os, "killpg", deny_group_kill)
    monkeypatch.setattr(
        sandbox, "_cleanup_command", lambda *_, **__: next(cleanup_results)
    )

    sandbox._terminate_container("docker", "sandbox-name", process)  # type: ignore[arg-type]

    assert process.killed


def test_cleanup_retries_a_transient_container_removal_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        pid = 123

        def poll(self) -> int:
            return 0

        def wait(self, *, timeout: float) -> int:
            assert timeout == 5
            return 0

    cleanup_results = iter(
        [
            subprocess.CompletedProcess([], 0),  # first rm request
            subprocess.CompletedProcess([], 0),  # still inspectable
            subprocess.CompletedProcess([], 0),  # second rm request
            subprocess.CompletedProcess([], 1),  # now absent
            subprocess.CompletedProcess([], 0),  # runtime healthy
        ]
    )
    waits: list[float] = []
    monkeypatch.setattr(sandbox.subprocess, "run", lambda *_, **__: None)
    monkeypatch.setattr(
        sandbox, "_cleanup_command", lambda *_, **__: next(cleanup_results)
    )
    monkeypatch.setattr(sandbox.time, "sleep", waits.append)

    sandbox._terminate_container(  # type: ignore[arg-type]
        "docker", "sandbox-name", Process()
    )

    assert waits == [0.05]


def test_unlisted_runtime_fails_closed_before_which_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DR_CODE_SANDBOX_RUNTIME", "podman-remote")

    def fail(_: str) -> None:
        raise AssertionError("which must not run for an unlisted runtime")

    monkeypatch.setattr(sandbox.shutil, "which", fail)

    with pytest.raises(sandbox.SandboxError, match="docker.*podman"):
        sandbox._resolve_runtime()
