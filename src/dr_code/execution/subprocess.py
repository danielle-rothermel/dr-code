"""Bounded execution of commands in fresh process groups."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import BinaryIO, Final, Protocol


MAX_SUBPROCESS_INPUT_BYTES: Final[int] = 4 * 1024 * 1024
MAX_SUBPROCESS_OUTPUT_BYTES: Final[int] = 1024 * 1024
_READ_CHUNK_BYTES: Final[int] = 64 * 1024
_POLL_INTERVAL_SECONDS: Final[float] = 0.01
_TERMINATION_TIMEOUT_SECONDS: Final[float] = 5.0
_IPC_THREAD_JOIN_SECONDS: Final[float] = 1.0


class SubprocessError(RuntimeError):
    """A bounded command did not complete normally."""


class SubprocessTimeoutError(SubprocessError):
    """The process exceeded its wall-clock deadline."""


class SubprocessOutputLimitError(SubprocessError):
    """The process exceeded the shared stdout and stderr limit."""


class SubprocessInfrastructureError(SubprocessError):
    """The host could not start, communicate with, or clean up the process."""


class SubprocessStartError(SubprocessInfrastructureError):
    """The host could not start the requested command."""


@dataclass(frozen=True, slots=True)
class SubprocessCompletedProcess:
    """Captured output and status from a completed command."""

    returncode: int
    stdout: str
    stderr: str


class PythonSubprocessRunner(Protocol):
    """Callable contract for executing one Python source program."""

    def __call__(
        self,
        *,
        source: str,
        input_text: str,
        timeout_seconds: float,
    ) -> SubprocessCompletedProcess: ...


@dataclass(slots=True)
class _BoundedOutput:
    stdout: bytearray
    stderr: bytearray
    size: int
    overflow: threading.Event
    lock: threading.Lock

    @classmethod
    def create(cls) -> _BoundedOutput:
        return cls(
            stdout=bytearray(),
            stderr=bytearray(),
            size=0,
            overflow=threading.Event(),
            lock=threading.Lock(),
        )


def run_python_subprocess(
    *,
    source: str,
    input_text: str,
    timeout_seconds: float,
) -> SubprocessCompletedProcess:
    """Execute source with bounded text IPC in isolated interpreter mode.

    ``-I`` isolates Python configuration and the child receives a minimal
    environment. It does not restrict the child's access to the invoking
    user's filesystem, processes, credentials, or network.
    """
    return run_subprocess(
        command=(sys.executable, "-I", "-c", source),
        input_text=input_text,
        timeout_seconds=timeout_seconds,
        environment=_child_environment(),
    )


def run_subprocess(
    *,
    command: Sequence[str],
    input_text: str,
    timeout_seconds: float,
    environment: Mapping[str, str] | None = None,
) -> SubprocessCompletedProcess:
    """Execute a command with bounded text IPC and a wall-clock deadline.

    ``environment=None`` inherits the invoking process environment. A supplied
    mapping replaces it. The command receives no filesystem, credential,
    process, or network isolation.
    """
    validated_command = _validate_command(command)
    validated_environment = _validate_environment(environment)
    if not isinstance(input_text, str):
        raise SubprocessError("subprocess input must be text")
    payload = input_text.encode("utf-8")
    if len(payload) > MAX_SUBPROCESS_INPUT_BYTES:
        raise SubprocessError(
            f"subprocess input exceeded {MAX_SUBPROCESS_INPUT_BYTES} bytes"
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int | float)
        or timeout_seconds <= 0
        or not math.isfinite(timeout_seconds)
    ):
        raise SubprocessError("subprocess timeout must be finite and positive")

    try:
        process = subprocess.Popen(
            validated_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=validated_environment,
        )
    except OSError as exc:
        raise SubprocessStartError("could not start subprocess") from exc

    output = _BoundedOutput.create()
    io_errors: list[BaseException] = []
    threads = [
        threading.Thread(
            target=_write_input,
            args=(process.stdin, payload, io_errors),
            daemon=True,
        ),
        threading.Thread(
            target=_read_bounded,
            args=(process.stdout, output.stdout, output, io_errors),
            daemon=True,
        ),
        threading.Thread(
            target=_read_bounded,
            args=(process.stderr, output.stderr, output, io_errors),
            daemon=True,
        ),
    ]
    started_threads: list[threading.Thread] = []
    timed_out = False
    try:
        for thread in threads:
            thread.start()
            started_threads.append(thread)

        deadline = time.monotonic() + timeout_seconds
        while process.poll() is None:
            if output.overflow.is_set():
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(_POLL_INTERVAL_SECONDS)
    finally:
        active_exception = sys.exception()
        try:
            _terminate_process_group(process)
        except SubprocessInfrastructureError as exc:
            if active_exception is not None:
                raise exc from active_exception
            raise
        finally:
            for thread in started_threads:
                thread.join(timeout=_IPC_THREAD_JOIN_SECONDS)

    if any(thread.is_alive() for thread in started_threads):
        raise SubprocessInfrastructureError(
            "subprocess IPC threads could not be terminated"
        )
    if timed_out:
        raise SubprocessTimeoutError(
            f"subprocess exceeded {timeout_seconds} seconds"
        )
    if output.overflow.is_set():
        raise SubprocessOutputLimitError(
            f"subprocess output exceeded {MAX_SUBPROCESS_OUTPUT_BYTES} bytes"
        )
    if io_errors:
        raise SubprocessInfrastructureError(
            "subprocess IPC failed"
        ) from io_errors[0]
    return SubprocessCompletedProcess(
        returncode=process.returncode,
        stdout=output.stdout.decode("utf-8", errors="replace"),
        stderr=output.stderr.decode("utf-8", errors="replace"),
    )


def _validate_command(command: Sequence[str]) -> tuple[str, ...]:
    if (
        isinstance(command, str | bytes)
        or not isinstance(command, Sequence)
        or not command
        or any(
            not isinstance(argument, str) or "\0" in argument
            for argument in command
        )
    ):
        raise SubprocessError(
            "subprocess command must be a nonempty sequence of strings"
        )
    if not command[0]:
        raise SubprocessError("subprocess executable must not be empty")
    return tuple(command)


def _validate_environment(
    environment: Mapping[str, str] | None,
) -> dict[str, str] | None:
    if environment is None:
        return None
    if not isinstance(environment, Mapping):
        raise SubprocessError(
            "subprocess environment must be a string mapping or None"
        )
    if any(
        not isinstance(key, str)
        or not key
        or "=" in key
        or "\0" in key
        or not isinstance(value, str)
        or "\0" in value
        for key, value in environment.items()
    ):
        raise SubprocessError(
            "subprocess environment must contain valid string entries"
        )
    return dict(environment)


def _child_environment() -> dict[str, str]:
    return {"OPENBLAS_NUM_THREADS": "1"}


def _write_input(
    stream: BinaryIO | None,
    payload: bytes,
    errors: list[BaseException],
) -> None:
    if stream is None:
        return
    try:
        stream.write(payload)
        stream.close()
    except BrokenPipeError:
        return
    except (OSError, ValueError) as exc:
        errors.append(exc)


def _read_bounded(
    stream: BinaryIO | None,
    destination: bytearray,
    output: _BoundedOutput,
    errors: list[BaseException],
) -> None:
    if stream is None:
        return
    try:
        while chunk := stream.read(_READ_CHUNK_BYTES):
            with output.lock:
                remaining = MAX_SUBPROCESS_OUTPUT_BYTES - output.size
                destination.extend(chunk[:remaining])
                output.size += min(len(chunk), remaining)
                if len(chunk) > remaining:
                    output.overflow.set()
                    return
    except (OSError, ValueError) as exc:
        errors.append(exc)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    signaling_error = _signal_process_group(process.pid)
    if signaling_error is not None and process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        except OSError as kill_error:
            signaling_error = kill_error

    try:
        process.wait(timeout=_TERMINATION_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise SubprocessInfrastructureError(
            "subprocess process group could not be terminated"
        ) from exc
    if signaling_error is None:
        return

    # Completion may win between the liveness poll and Popen.kill(). Retry the
    # group after reaping the leader so descendants cannot survive that race.
    # A normally exited leader makes the first error stale; live pipe-reader
    # threads remain the descendant-cleanup backstop.
    remaining_error = _signal_process_group(process.pid)
    if remaining_error is None or process.returncode != -signal.SIGKILL:
        return
    raise _process_group_signaling_error(remaining_error) from remaining_error


def _signal_process_group(process_group_id: int) -> OSError | None:
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        return None
    except OSError as exc:
        return exc
    return None


def _process_group_signaling_error(
    error: OSError,
) -> SubprocessInfrastructureError:
    return SubprocessInfrastructureError(
        "subprocess process group could not be signaled: "
        f"errno={error.errno} ({error.strerror or str(error)})"
    )
