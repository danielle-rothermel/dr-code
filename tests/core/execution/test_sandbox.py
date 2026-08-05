"""Deterministic checks for the public OCI sandbox failure contract.

The runtime helpers remain private because the production boundary is the
three-keyword ``run_python_in_sandbox`` callable. Tests use state-controlled
process, stream, and thread doubles to force failures that a live runtime
cannot reproduce reliably.
"""

from __future__ import annotations

import io
import subprocess
from collections.abc import Callable
from typing import BinaryIO

import pytest

from dr_code.core.execution import sandbox


class _RuntimeLookupReached(Exception):
    pass


class _FailingStream:
    def __init__(self, error: OSError) -> None:
        self._error = error

    def read(self, _: int = -1) -> bytes:
        raise self._error

    def write(self, _: bytes) -> int:
        raise self._error

    def close(self) -> None:
        pass


class _FakeProcess:
    def __init__(
        self,
        *,
        stdin: BinaryIO | None = None,
        stdout: BinaryIO | None = None,
        stderr: BinaryIO | None = None,
        running: bool = True,
    ) -> None:
        self.stdin = stdin if stdin is not None else io.BytesIO()
        self.stdout = stdout if stdout is not None else io.BytesIO()
        self.stderr = stderr if stderr is not None else io.BytesIO()
        self.pid = 1234
        self.returncode: int | None = None if running else 0

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float) -> int:
        del timeout
        self.returncode = -9
        return self.returncode


class _SynchronousThread:
    def __init__(
        self,
        *,
        target: Callable[..., None],
        args: tuple[object, ...],
        daemon: bool,
    ) -> None:
        del daemon
        self._target = target
        self._args = args

    def start(self) -> None:
        self._target(*self._args)

    def join(self, timeout: float) -> None:
        del timeout

    def is_alive(self) -> bool:
        return False


class _LiveThread(_SynchronousThread):
    def start(self) -> None:
        pass

    def is_alive(self) -> bool:
        return True


class _UnstartableThread(_SynchronousThread):
    error = RuntimeError("thread capacity exhausted")

    def start(self) -> None:
        raise self.error


def _prepare_public_run(
    monkeypatch: pytest.MonkeyPatch,
    process: _FakeProcess,
) -> None:
    monkeypatch.setattr(sandbox, "_resolve_runtime", lambda: "/docker")
    monkeypatch.setattr(sandbox, "_require_local_image", lambda *_: None)
    monkeypatch.setattr(sandbox.subprocess, "Popen", lambda *_, **__: process)


def _completed(returncode: int) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess([], returncode)


def _run_with_timeout() -> sandbox.SandboxCompletedProcess:
    return sandbox.run_python_in_sandbox(
        source="input()",
        input_json="{}",
        timeout_seconds=0.5,
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


def test_hung_image_inspection_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def hang(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="docker", timeout=10)

    monkeypatch.setattr(sandbox.subprocess, "run", hang)

    with pytest.raises(sandbox.SandboxError, match="image inspection failed"):
        sandbox._require_local_image("docker", sandbox.SANDBOX_IMAGE)


def test_missing_docker_runtime_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox.shutil, "which", lambda _: None)

    with pytest.raises(sandbox.SandboxError, match="unavailable: docker"):
        sandbox._resolve_runtime()


def test_runtime_resolution_uses_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []

    def which(name: str) -> str:
        requested.append(name)
        return "/usr/bin/docker"

    monkeypatch.setattr(sandbox.shutil, "which", which)

    assert sandbox._resolve_runtime() == "/usr/bin/docker"
    assert requested == ["docker"]


def test_input_at_exact_utf8_byte_limit_reaches_runtime_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reached() -> str:
        raise _RuntimeLookupReached

    monkeypatch.setattr(sandbox, "_resolve_runtime", reached)

    exact_limit = "é" * (sandbox.MAX_SANDBOX_INPUT_BYTES // 2)
    with pytest.raises(_RuntimeLookupReached):
        sandbox.run_python_in_sandbox(
            source="pass", input_json=exact_limit, timeout_seconds=1.0
        )


def test_input_over_utf8_byte_limit_fails_before_runtime_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected() -> str:
        raise AssertionError("runtime lookup must not be reached")

    monkeypatch.setattr(sandbox, "_resolve_runtime", unexpected)

    over_limit = "é" * (sandbox.MAX_SANDBOX_INPUT_BYTES // 2 + 1)
    with pytest.raises(sandbox.SandboxError, match="input exceeded"):
        sandbox.run_python_in_sandbox(
            source="pass", input_json=over_limit, timeout_seconds=1.0
        )


def test_non_utf8_input_fails_at_the_sandbox_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sandbox,
        "_resolve_runtime",
        lambda: pytest.fail("runtime lookup must not be reached"),
    )

    with pytest.raises(sandbox.SandboxError, match="valid UTF-8") as raised:
        sandbox.run_python_in_sandbox(
            source="pass", input_json="\ud800", timeout_seconds=1.0
        )

    assert isinstance(raised.value.__cause__, UnicodeEncodeError)


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_timeout_fails_before_runtime_lookup(
    monkeypatch: pytest.MonkeyPatch,
    timeout: float,
) -> None:
    monkeypatch.setattr(
        sandbox,
        "_resolve_runtime",
        lambda: pytest.fail("runtime lookup must not be reached"),
    )

    with pytest.raises(sandbox.SandboxError, match="finite and positive"):
        sandbox.run_python_in_sandbox(
            source="pass", input_json="{}", timeout_seconds=timeout
        )


def test_process_startup_failure_is_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_error = OSError("exec failed")
    monkeypatch.setattr(sandbox, "_resolve_runtime", lambda: "/docker")
    monkeypatch.setattr(sandbox, "_require_local_image", lambda *_: None)

    def fail(*args: object, **kwargs: object) -> None:
        raise startup_error

    monkeypatch.setattr(sandbox.subprocess, "Popen", fail)

    with pytest.raises(
        sandbox.SandboxError, match="failed to start"
    ) as raised:
        _run_with_timeout()

    assert raised.value.__cause__ is startup_error


def test_ipc_thread_startup_failure_cleans_up_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    _prepare_public_run(monkeypatch, process)
    monkeypatch.setattr(sandbox.threading, "Thread", _UnstartableThread)
    cleanup_called = False

    def terminate(*_: object) -> None:
        nonlocal cleanup_called
        cleanup_called = True
        process.returncode = -9

    monkeypatch.setattr(sandbox, "_terminate_container", terminate)

    with pytest.raises(
        sandbox.SandboxError, match="threads failed to start"
    ) as raised:
        _run_with_timeout()

    assert cleanup_called is True
    assert raised.value.__cause__ is _UnstartableThread.error


@pytest.mark.parametrize("failing_stream", ["stdin", "stdout", "stderr"])
def test_ipc_failure_terminates_the_sandbox_and_preserves_its_cause(
    monkeypatch: pytest.MonkeyPatch,
    failing_stream: str,
) -> None:
    ipc_error = OSError(f"{failing_stream} failed")
    streams: dict[str, BinaryIO] = {
        "stdin": io.BytesIO(),
        "stdout": io.BytesIO(),
        "stderr": io.BytesIO(),
    }
    streams[failing_stream] = _FailingStream(ipc_error)  # type: ignore[assignment]
    process = _FakeProcess(**streams)
    _prepare_public_run(monkeypatch, process)
    monkeypatch.setattr(sandbox.threading, "Thread", _SynchronousThread)
    cleanup_called = False

    def terminate(*_: object) -> None:
        nonlocal cleanup_called
        cleanup_called = True
        process.returncode = -9

    monkeypatch.setattr(sandbox, "_terminate_container", terminate)

    with pytest.raises(sandbox.SandboxError, match="IPC failed") as raised:
        _run_with_timeout()

    assert cleanup_called is True
    assert raised.value.__cause__ is ipc_error


def test_stdout_overflow_uses_the_bounded_public_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(
        stdout=io.BytesIO(b"x" * (sandbox.MAX_SANDBOX_OUTPUT_BYTES + 1))
    )
    _prepare_public_run(monkeypatch, process)
    monkeypatch.setattr(sandbox.threading, "Thread", _SynchronousThread)

    def terminate(*_: object) -> None:
        process.returncode = -9

    monkeypatch.setattr(sandbox, "_terminate_container", terminate)

    with pytest.raises(sandbox.SandboxOutputLimitError):
        _run_with_timeout()


def test_live_ipc_thread_after_join_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(running=False)
    _prepare_public_run(monkeypatch, process)
    monkeypatch.setattr(sandbox.threading, "Thread", _LiveThread)

    with pytest.raises(sandbox.SandboxError, match="threads.*terminated"):
        _run_with_timeout()


def test_timeout_failure_survives_successful_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    _prepare_public_run(monkeypatch, process)
    monkeypatch.setattr(sandbox.threading, "Thread", _SynchronousThread)
    monotonic = iter((0.0, 1.0))
    monkeypatch.setattr(sandbox.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(
        sandbox.subprocess,
        "run",
        lambda command, **_: _completed(
            1 if command[1:3] == ["container", "inspect"] else 0
        ),
    )
    monkeypatch.setattr(sandbox.os, "killpg", lambda *_: None)

    with pytest.raises(sandbox.SandboxTimeoutError):
        _run_with_timeout()


def test_process_group_signal_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    _prepare_public_run(monkeypatch, process)
    monkeypatch.setattr(sandbox.threading, "Thread", _SynchronousThread)
    monotonic = iter((0.0, 1.0))
    monkeypatch.setattr(sandbox.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(
        sandbox.subprocess,
        "run",
        lambda command, **_: _completed(
            1 if command[1:3] == ["container", "inspect"] else 0
        ),
    )
    signal_error = PermissionError("signal denied")

    def fail_signal(*_: object) -> None:
        raise signal_error

    monkeypatch.setattr(sandbox.os, "killpg", fail_signal)

    with pytest.raises(
        sandbox.SandboxError, match="process group could not be terminated"
    ) as raised:
        _run_with_timeout()

    assert isinstance(raised.value.__cause__, sandbox.SandboxTimeoutError)
    assert raised.value.__cause__.__cause__ is signal_error


def test_surviving_container_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    _prepare_public_run(monkeypatch, process)
    monkeypatch.setattr(sandbox.threading, "Thread", _SynchronousThread)
    monotonic = iter((0.0, 1.0))
    monkeypatch.setattr(sandbox.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(
        sandbox.subprocess, "run", lambda *_, **__: _completed(0)
    )
    monkeypatch.setattr(sandbox.os, "killpg", lambda *_: None)

    with pytest.raises(sandbox.SandboxError, match="container survived"):
        _run_with_timeout()


def test_runtime_unavailable_during_cleanup_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    _prepare_public_run(monkeypatch, process)
    monkeypatch.setattr(sandbox.threading, "Thread", _SynchronousThread)
    monotonic = iter((0.0, 1.0))
    monkeypatch.setattr(sandbox.time, "monotonic", lambda: next(monotonic))

    def runtime_result(command: list[str], **_: object) -> object:
        if command[1:3] == ["container", "inspect"]:
            return _completed(1)
        if command[1:] == ["info"]:
            return _completed(1)
        return _completed(0)

    monkeypatch.setattr(sandbox.subprocess, "run", runtime_result)
    monkeypatch.setattr(sandbox.os, "killpg", lambda *_: None)

    with pytest.raises(
        sandbox.SandboxError,
        match="runtime could not confirm container termination",
    ):
        _run_with_timeout()
