"""Bounded execution of commands in fresh process groups."""

from __future__ import annotations

import io
import math
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import BinaryIO, Final, Protocol


MAX_SUBPROCESS_INPUT_BYTES: Final[int] = 4 * 1024 * 1024
MAX_SUBPROCESS_OUTPUT_BYTES: Final[int] = 1024 * 1024
_READ_CHUNK_BYTES: Final[int] = 64 * 1024
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

    @classmethod
    def create(cls) -> _BoundedOutput:
        return cls(
            stdout=bytearray(),
            stderr=bytearray(),
            size=0,
        )


@dataclass(slots=True)
class _ExecutionMonitor:
    condition: threading.Condition
    leader_exit_observed: bool = False
    leader_reaped: bool = False
    returncode: int | None = None
    output_overflow: bool = False
    timeout_selected: bool = False
    io_error: BaseException | None = None
    monitor_error: BaseException | None = None
    reaper_error: BaseException | None = None

    @classmethod
    def create(cls) -> _ExecutionMonitor:
        return cls(condition=threading.Condition())

    def record_process_exit(self, returncode: int) -> None:
        with self.condition:
            self.leader_exit_observed = True
            self.leader_reaped = True
            self.returncode = returncode
            self.condition.notify_all()

    def record_reaper_error(self, error: BaseException) -> None:
        with self.condition:
            if self.reaper_error is None:
                self.reaper_error = error
            self.condition.notify_all()

    def record_output_overflow(self) -> None:
        with self.condition:
            self.output_overflow = True
            self.condition.notify_all()

    def append_output(
        self,
        destination: bytearray,
        output: _BoundedOutput,
        chunk: bytes,
    ) -> bool:
        with self.condition:
            remaining = MAX_SUBPROCESS_OUTPUT_BYTES - output.size
            destination.extend(chunk[:remaining])
            output.size += min(len(chunk), remaining)
            if len(chunk) <= remaining:
                return True
            self.output_overflow = True
            self.condition.notify_all()
            return False

    def record_io_error(self, error: BaseException) -> None:
        with self.condition:
            if self.io_error is None:
                self.io_error = error
            self.condition.notify_all()

    def record_monitor_error(self, error: BaseException) -> None:
        with self.condition:
            if self.monitor_error is None:
                self.monitor_error = error
            self.condition.notify_all()

    def wait_for_terminal(
        self, process_id: int, timeout_seconds: float
    ) -> None:
        """Select the first execution trigger at the process boundary."""
        try:
            with self.condition:
                self._wait_for_trigger_or_expiry(
                    process_id,
                    timeout_seconds,
                )
        except Exception as exc:
            self.record_monitor_error(exc)

    def _wait_for_trigger_or_expiry(
        self,
        process_id: int,
        timeout_seconds: float,
    ) -> None:
        if timeout_seconds <= threading.TIMEOUT_MAX:
            if self.condition.wait_for(
                self._has_execution_trigger,
                timeout_seconds,
            ):
                return
            self._select_at_logical_expiry(process_id)
            return

        if isinstance(timeout_seconds, float):
            remaining = Fraction.from_float(timeout_seconds)
        else:
            remaining = Fraction(timeout_seconds)
        maximum_chunk = Fraction.from_float(threading.TIMEOUT_MAX)
        while remaining > maximum_chunk:
            if self.condition.wait_for(
                self._has_execution_trigger,
                threading.TIMEOUT_MAX,
            ):
                return
            remaining -= maximum_chunk

        if self.condition.wait_for(
            self._has_execution_trigger,
            float(remaining),
        ):
            return
        self._select_at_logical_expiry(process_id)

    def _has_execution_trigger(self) -> bool:
        return (
            self.leader_exit_observed
            or self.output_overflow
            or self.timeout_selected
            or self.io_error is not None
            or self.monitor_error is not None
            or self.reaper_error is not None
        )

    def _select_at_logical_expiry(self, process_id: int) -> None:
        if self._has_execution_trigger():
            return
        try:
            status = os.waitid(
                os.P_PID,
                process_id,
                os.WEXITED | os.WNOHANG | os.WNOWAIT,
            )
        except ChildProcessError:
            self.leader_exit_observed = True
        except OSError as exc:
            self.monitor_error = exc
        else:
            if status is None:
                self.timeout_selected = True
            else:
                self.leader_exit_observed = True
        self.condition.notify_all()

    def wait_for_reaping(self, timeout_seconds: float) -> bool:
        with self.condition:
            self.condition.wait_for(
                lambda: self.leader_reaped or self.reaper_error is not None,
                timeout_seconds,
            )
            return self.leader_reaped

    def get_reaper_error(self) -> BaseException | None:
        with self.condition:
            return self.reaper_error

    def returncode_or_raise(self, timeout_seconds: float) -> int:
        with self.condition:
            monitor_error = self.monitor_error
            reaper_error = self.reaper_error
            timeout_selected = self.timeout_selected
            output_overflow = self.output_overflow
            io_error = self.io_error
            returncode = self.returncode

        if monitor_error is not None:
            raise SubprocessInfrastructureError(
                "subprocess execution monitoring failed"
            ) from monitor_error
        if reaper_error is not None:
            raise SubprocessInfrastructureError(
                "subprocess process group could not be terminated"
            ) from reaper_error
        if timeout_selected:
            raise SubprocessTimeoutError(
                f"subprocess exceeded {timeout_seconds} seconds"
            )
        if output_overflow:
            raise SubprocessOutputLimitError(
                "subprocess output exceeded "
                f"{MAX_SUBPROCESS_OUTPUT_BYTES} bytes"
            )
        if io_error is not None:
            raise SubprocessInfrastructureError(
                "subprocess IPC failed"
            ) from io_error
        if returncode is None:
            raise SubprocessInfrastructureError(
                "subprocess process group could not be terminated"
            )
        return returncode


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
    monitor = _ExecutionMonitor.create()
    process_waiter = threading.Thread(
        target=_wait_for_process,
        args=(process, monitor),
        daemon=True,
        name="dr-code-subprocess-waiter",
    )
    ipc_threads = [
        threading.Thread(
            target=_write_input,
            args=(process.stdin, payload, monitor),
            daemon=True,
            name="dr-code-subprocess-stdin",
        ),
        threading.Thread(
            target=_read_bounded,
            args=(process.stdout, output.stdout, output, monitor),
            daemon=True,
            name="dr-code-subprocess-stdout",
        ),
        threading.Thread(
            target=_read_bounded,
            args=(process.stderr, output.stderr, output, monitor),
            daemon=True,
            name="dr-code-subprocess-stderr",
        ),
    ]
    started_process_waiter: threading.Thread | None = None
    started_ipc_threads: list[threading.Thread] = []
    pending_error: BaseException | None = None
    try:
        process_waiter.start()
        started_process_waiter = process_waiter
        for thread in ipc_threads:
            thread.start()
            started_ipc_threads.append(thread)
        monitor.wait_for_terminal(process.pid, timeout_seconds)
    except Exception as exc:
        monitor.record_monitor_error(exc)
    except BaseException as exc:
        pending_error = exc

    cleanup_error: SubprocessInfrastructureError | None = None
    try:
        _terminate_process_group(
            process,
            monitor,
            process_waiter=started_process_waiter,
            ipc_threads=started_ipc_threads,
        )
    except SubprocessInfrastructureError as exc:
        cleanup_error = exc

    if cleanup_error is not None:
        if pending_error is not None:
            raise cleanup_error from pending_error
        raise cleanup_error
    if pending_error is not None:
        raise pending_error

    returncode = monitor.returncode_or_raise(timeout_seconds)
    return SubprocessCompletedProcess(
        returncode=returncode,
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
    monitor: _ExecutionMonitor,
) -> None:
    if stream is None:
        return
    try:
        stream.write(payload)
        stream.close()
    except BrokenPipeError:
        return
    except (OSError, ValueError) as exc:
        monitor.record_io_error(exc)


def _read_bounded(
    stream: BinaryIO | None,
    destination: bytearray,
    output: _BoundedOutput,
    monitor: _ExecutionMonitor,
) -> None:
    if stream is None:
        return
    try:
        if isinstance(stream, io.BufferedIOBase):
            read_chunk = stream.read1
        else:
            read_chunk = stream.read
        while chunk := read_chunk(_READ_CHUNK_BYTES):
            if not monitor.append_output(destination, output, chunk):
                return
    except (OSError, ValueError) as exc:
        monitor.record_io_error(exc)


def _wait_for_process(
    process: subprocess.Popen[bytes],
    monitor: _ExecutionMonitor,
) -> None:
    try:
        returncode = process.wait()
    except Exception as exc:
        monitor.record_reaper_error(exc)
    else:
        monitor.record_process_exit(returncode)


def _terminate_process_group(
    process: subprocess.Popen[bytes],
    monitor: _ExecutionMonitor,
    *,
    process_waiter: threading.Thread | None,
    ipc_threads: Sequence[threading.Thread],
) -> None:
    failures: list[
        tuple[SubprocessInfrastructureError, BaseException | None]
    ] = []

    signaling_error = _signal_process_group(process.pid)
    if signaling_error is not None and process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        except OSError as kill_error:
            failures.append((_process_termination_error(), kill_error))

    reaped = False
    if process_waiter is not None:
        try:
            reaped = monitor.wait_for_reaping(_TERMINATION_TIMEOUT_SECONDS)
        except Exception as exc:
            failures.append((_process_termination_error(), exc))
    if not reaped and not failures:
        failures.append(
            (_process_termination_error(), monitor.get_reaper_error())
        )

    # The leader may exit just before the first group signal. Signaling again
    # after its reap closes that race for descendants that retain the group.
    remaining_signaling_error = _signal_process_group(process.pid)
    if remaining_signaling_error is not None:
        failures.append(
            (
                _process_group_signaling_error(remaining_signaling_error),
                remaining_signaling_error,
            )
        )

    if process_waiter is not None:
        try:
            process_waiter.join(timeout=_IPC_THREAD_JOIN_SECONDS)
        except Exception as exc:
            failures.append((_process_waiter_join_error(), exc))
        else:
            if process_waiter.is_alive():
                failures.append((_process_waiter_join_error(), None))

    ipc_join_error: BaseException | None = None
    ipc_thread_alive = False
    for thread in ipc_threads:
        try:
            thread.join(timeout=_IPC_THREAD_JOIN_SECONDS)
        except Exception as exc:
            if ipc_join_error is None:
                ipc_join_error = exc
        else:
            ipc_thread_alive = ipc_thread_alive or thread.is_alive()
    if ipc_join_error is not None or ipc_thread_alive:
        failures.append(
            (
                SubprocessInfrastructureError(
                    "subprocess IPC threads could not be terminated"
                ),
                ipc_join_error,
            )
        )

    if not failures:
        return
    error, cause = failures[0]
    if cause is None:
        raise error
    raise error from cause


def _process_termination_error() -> SubprocessInfrastructureError:
    return SubprocessInfrastructureError(
        "subprocess process group could not be terminated"
    )


def _process_waiter_join_error() -> SubprocessInfrastructureError:
    return SubprocessInfrastructureError(
        "subprocess process waiter thread could not be terminated"
    )


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
