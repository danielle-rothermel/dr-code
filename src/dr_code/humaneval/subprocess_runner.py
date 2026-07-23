"""Bounded host-subprocess execution for HumanEval runner programs."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import BinaryIO, Final, Protocol


# HumanEval+/113 alone serializes to 1,255,579 bytes. Four MiB admits that
# payload and escaped candidate source while keeping host IPC input bounded.
MAX_SUBPROCESS_INPUT_BYTES: Final[int] = 4 * 1024 * 1024
MAX_SUBPROCESS_OUTPUT_BYTES: Final[int] = 1_048_576
CANDIDATE_KILL_RETURNCODES: Final[frozenset[int]] = frozenset(
    {-int(signal.SIGKILL), -int(signal.SIGSEGV)}
)
_READ_CHUNK_BYTES: Final[int] = 65_536
_POLL_INTERVAL_SECONDS: Final[float] = 0.01
_TERMINATION_TIMEOUT_SECONDS: Final[float] = 5.0


class SubprocessError(RuntimeError):
    """The trusted host-subprocess boundary could not complete execution."""


class SubprocessTimeoutError(SubprocessError):
    """The subprocess exceeded its wall-clock deadline."""


class SubprocessOutputLimitError(SubprocessError):
    """The subprocess emitted more data than the IPC contract permits."""


@dataclass(frozen=True, slots=True)
class SubprocessCompletedProcess:
    returncode: int
    stdout: str
    stderr: str


class SubprocessRunner(Protocol):
    """Callable contract for executing one Python runner program."""

    def __call__(
        self,
        *,
        source: str,
        input_json: str,
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
    input_json: str,
    timeout_seconds: float,
) -> SubprocessCompletedProcess:
    """Run one isolated Python child using bounded JSON stdin/stdout IPC.

    The child is isolated from the caller's Python state by a fresh
    ``sys.executable`` process and ``-I`` interpreter mode. It is not a
    security sandbox: candidate code retains the host user's filesystem and
    network permissions.
    """
    payload = input_json.encode("utf-8")
    if len(payload) > MAX_SUBPROCESS_INPUT_BYTES:
        raise SubprocessError(
            f"subprocess input exceeded {MAX_SUBPROCESS_INPUT_BYTES} bytes"
        )
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise SubprocessError("subprocess timeout must be finite and positive")

    try:
        process = subprocess.Popen(
            [sys.executable, "-I", "-c", source],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=_child_environment(),
        )
    except OSError as exc:
        raise SubprocessError("could not start Python subprocess") from exc

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
        except SubprocessError as exc:
            if active_exception is not None:
                raise exc from active_exception
            raise
        finally:
            for thread in started_threads:
                thread.join(timeout=1.0)

    if any(thread.is_alive() for thread in started_threads):
        raise SubprocessError("subprocess IPC threads could not be terminated")
    if timed_out:
        raise SubprocessTimeoutError(
            f"subprocess exceeded {timeout_seconds} seconds"
        )
    if output.overflow.is_set():
        raise SubprocessOutputLimitError(
            f"subprocess output exceeded {MAX_SUBPROCESS_OUTPUT_BYTES} bytes"
        )
    if io_errors:
        raise SubprocessError("subprocess IPC failed") from io_errors[0]
    return SubprocessCompletedProcess(
        returncode=process.returncode,
        stdout=output.stdout.decode("utf-8", errors="replace"),
        stderr=output.stderr.decode("utf-8", errors="replace"),
    )


def _child_environment() -> dict[str, str]:
    return {
        "OPENBLAS_NUM_THREADS": "1",
    }


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
    signaling_error: OSError | None = None
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        # ``poll`` reaps a normally completed group leader before cleanup.
        # Keep trying ``killpg`` so descendants cannot outlive that leader,
        # but a signaling error is actionable only while the direct child is
        # still live. On macOS, treating post-reap errors as live-process
        # failures produced rare false infrastructure failures under churn.
        if process.poll() is None:
            signaling_error = exc
            try:
                process.kill()
            except ProcessLookupError:
                signaling_error = None
            except OSError as kill_error:
                raise _process_group_signaling_error(kill_error) from exc

    try:
        process.wait(timeout=_TERMINATION_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise SubprocessError(
            "subprocess process group could not be terminated"
        ) from exc
    if signaling_error is not None:
        raise _process_group_signaling_error(
            signaling_error
        ) from signaling_error


def _process_group_signaling_error(error: OSError) -> SubprocessError:
    return SubprocessError(
        "subprocess process group could not be signaled: "
        f"errno={error.errno} ({error.strerror or str(error)})"
    )
