from __future__ import annotations

import concurrent.futures
import errno
import io
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from dr_code.execution import subprocess as subprocess_execution
from dr_code.execution.subprocess import (
    MAX_SUBPROCESS_INPUT_BYTES,
    MAX_SUBPROCESS_OUTPUT_BYTES,
    SubprocessCompletedProcess,
    SubprocessError,
    SubprocessInfrastructureError,
    SubprocessOutputLimitError,
    SubprocessStartError,
    SubprocessTimeoutError,
    run_python_subprocess,
    run_subprocess,
)


def test_execution_package_imports_without_humaneval_dependencies() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import json, sys\n"
                "import dr_code.execution\n"
                "print(json.dumps(sorted(name for name in sys.modules "
                "if name.startswith(('dr_code.humaneval', 'pydantic')))))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(Path.cwd() / "src")},
    )

    assert json.loads(completed.stdout) == []


def test_subprocess_round_trips_text_in_a_fresh_python_child() -> None:
    first = run_python_subprocess(
        source=(
            "import json, os, sys\n"
            "payload = json.load(sys.stdin)\n"
            "print(json.dumps({'payload': payload, 'pid': os.getpid()}))\n"
        ),
        input_text='{"value": 7}',
        timeout_seconds=2.0,
    )
    second = run_python_subprocess(
        source="import json, os; print(json.dumps({'pid': os.getpid()}))",
        input_text="",
        timeout_seconds=2.0,
    )

    first_output = json.loads(first.stdout)
    second_output = json.loads(second.stdout)
    assert first.returncode == 0
    assert first.stderr == ""
    assert first_output["payload"] == {"value": 7}
    assert first_output["pid"] != os.getpid()
    assert first_output["pid"] != second_output["pid"]


def test_general_subprocess_runs_arbitrary_command_with_text_input() -> None:
    completed = run_subprocess(
        command=(
            sys.executable,
            "-c",
            "import sys; sys.stdout.write(sys.stdin.read().upper())",
        ),
        input_text="general command",
        timeout_seconds=2.0,
    )

    assert completed == SubprocessCompletedProcess(
        returncode=0,
        stdout="GENERAL COMMAND",
        stderr="",
    )


def test_general_subprocess_inherits_environment_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DR_CODE_INHERITED_ENV", "inherited")

    completed = run_subprocess(
        command=(
            sys.executable,
            "-c",
            "import os; print(os.environ['DR_CODE_INHERITED_ENV'])",
        ),
        input_text="",
        timeout_seconds=2.0,
    )

    assert completed.stdout == "inherited\n"


def test_general_subprocess_explicit_environment_replaces_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DR_CODE_INHERITED_ENV", "must-not-leak")

    completed = run_subprocess(
        command=(
            sys.executable,
            "-c",
            (
                "import json, os; "
                "print(json.dumps({"
                "'explicit': os.environ['DR_CODE_EXPLICIT_ENV'], "
                "'inherited': os.environ.get('DR_CODE_INHERITED_ENV')"
                "}))"
            ),
        ),
        input_text="",
        timeout_seconds=2.0,
        environment={"DR_CODE_EXPLICIT_ENV": "explicit"},
    )

    assert json.loads(completed.stdout) == {
        "explicit": "explicit",
        "inherited": None,
    }


def test_subprocess_uses_minimal_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DR_CODE_SHOULD_NOT_REACH_CHILD", "secret")

    completed = run_python_subprocess(
        source="import json, os; print(json.dumps(dict(os.environ)))",
        input_text="",
        timeout_seconds=2.0,
    )

    environment = json.loads(completed.stdout)
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
    assert "DR_CODE_SHOULD_NOT_REACH_CHILD" not in environment
    assert set(environment) <= {
        "LC_CTYPE",
        "OPENBLAS_NUM_THREADS",
        "__CF_USER_TEXT_ENCODING",
    }


def test_subprocess_invokes_isolated_sys_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def record_run(**kwargs: Any) -> SubprocessCompletedProcess:
        seen.update(kwargs)
        return SubprocessCompletedProcess(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess_execution, "run_subprocess", record_run)

    completed = run_python_subprocess(
        source="pass",
        input_text="input",
        timeout_seconds=1.0,
    )

    assert completed.returncode == 0
    assert seen == {
        "command": (sys.executable, "-I", "-c", "pass"),
        "input_text": "input",
        "timeout_seconds": 1.0,
        "environment": {"OPENBLAS_NUM_THREADS": "1"},
    }


def test_general_subprocess_starts_a_new_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fail_start(command: tuple[str, ...], **kwargs: Any) -> None:
        seen["command"] = command
        seen.update(kwargs)
        raise FileNotFoundError(errno.ENOENT, "No such file or directory")

    monkeypatch.setattr(subprocess, "Popen", fail_start)

    with pytest.raises(
        SubprocessStartError, match="could not start"
    ) as caught:
        run_subprocess(
            command=("missing-executable",),
            input_text="",
            timeout_seconds=1.0,
        )

    assert type(caught.value) is SubprocessStartError
    assert isinstance(caught.value.__cause__, FileNotFoundError)
    assert seen["command"] == ("missing-executable",)
    assert seen["start_new_session"] is True
    assert seen["env"] is None


def test_general_subprocess_returns_nonzero_result() -> None:
    completed = run_subprocess(
        command=(
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('failed'); raise SystemExit(17)",
        ),
        input_text="",
        timeout_seconds=2.0,
    )

    assert completed == SubprocessCompletedProcess(
        returncode=17,
        stdout="",
        stderr="failed",
    )


@pytest.mark.parametrize("timeout_seconds", [0.0, -1.0, math.inf, math.nan])
def test_subprocess_rejects_invalid_timeout(timeout_seconds: float) -> None:
    with pytest.raises(
        SubprocessError,
        match="timeout must be finite and positive",
    ):
        run_python_subprocess(
            source="pass",
            input_text="",
            timeout_seconds=timeout_seconds,
        )


def test_subprocess_accepts_exact_input_limit_without_deadlock() -> None:
    completed = run_subprocess(
        command=(
            sys.executable,
            "-c",
            "import sys\n"
            "sys.stdout.write('ready\\n')\n"
            "sys.stdout.flush()\n"
            "data = sys.stdin.buffer.read()\n"
            "sys.stdout.write(str(len(data)))\n",
        ),
        input_text="x" * MAX_SUBPROCESS_INPUT_BYTES,
        timeout_seconds=3.0,
    )

    assert completed.stdout == f"ready\n{MAX_SUBPROCESS_INPUT_BYTES}"


def test_subprocess_rejects_oversized_input_before_starting_child() -> None:
    with pytest.raises(SubprocessError, match="input exceeded"):
        run_subprocess(
            command=(sys.executable, "-c", "pass"),
            input_text="x" * (MAX_SUBPROCESS_INPUT_BYTES + 1),
            timeout_seconds=1.0,
        )


@pytest.mark.parametrize("stream", [1, 2], ids=["stdout", "stderr"])
def test_subprocess_stops_at_combined_output_limit(stream: int) -> None:
    with pytest.raises(SubprocessOutputLimitError, match="output exceeded"):
        run_python_subprocess(
            source=(
                "import os\n"
                f"os.write({stream}, b'x' * "
                f"{MAX_SUBPROCESS_OUTPUT_BYTES + 1})\n"
            ),
            input_text="",
            timeout_seconds=2.0,
        )


def test_subprocess_output_limit_is_shared_by_stdout_and_stderr() -> None:
    half_limit = MAX_SUBPROCESS_OUTPUT_BYTES // 2 + 1

    with pytest.raises(SubprocessOutputLimitError, match="output exceeded"):
        run_subprocess(
            command=(
                sys.executable,
                "-c",
                "import os\n"
                f"os.write(1, b'x' * {half_limit})\n"
                f"os.write(2, b'y' * {half_limit})\n",
            ),
            input_text="",
            timeout_seconds=2.0,
        )


def test_subprocess_output_overflow_closes_descendant_owned_channel() -> None:
    ready_frame = b"descendant-ready\n"
    with tempfile.TemporaryDirectory(
        prefix="dr-code-descendant-",
        dir="/tmp",
    ) as socket_directory:
        socket_path = str(Path(socket_directory) / "channel.sock")
        descendant_source = (
            "import os, socket, sys\n"
            "channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
            f"channel.connect({socket_path!r})\n"
            f"channel.sendall({ready_frame!r})\n"
            "trigger = channel.recv(1)\n"
            "if trigger != b'x': raise SystemExit('missing trigger')\n"
            "os.write(int(sys.argv[1]), b'r')\n"
            "os.close(int(sys.argv[1]))\n"
            "channel.recv(1)\n"
        )
        leader_source = (
            "import os, subprocess, sys\n"
            "read_fd, write_fd = os.pipe()\n"
            "hold_read_fd, hold_write_fd = os.pipe()\n"
            f"descendant_source = {descendant_source!r}\n"
            "subprocess.Popen(\n"
            "    [sys.executable, '-c', descendant_source, str(write_fd)],\n"
            "    pass_fds=(write_fd,),\n"
            ")\n"
            "os.close(write_fd)\n"
            "if os.read(read_fd, 1) != b'r': raise SystemExit('not ready')\n"
            "os.close(read_fd)\n"
            f"remaining = {MAX_SUBPROCESS_OUTPUT_BYTES + 1}\n"
            "chunk = b'x' * 65536\n"
            "while remaining:\n"
            "    written = os.write(1, chunk[:remaining])\n"
            "    remaining -= written\n"
            "os.read(hold_read_fd, 1)\n"
        )

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(socket_path)
            listener.listen(1)
            listener.settimeout(5.0)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=1
            ) as executor:
                future = executor.submit(
                    run_subprocess,
                    command=(sys.executable, "-c", leader_source),
                    input_text="",
                    timeout_seconds=5.0,
                )
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(5.0)
                    assert _recv_exact(connection, len(ready_frame)) == (
                        ready_frame
                    )
                    connection.sendall(b"x")

                    with pytest.raises(
                        SubprocessOutputLimitError,
                        match="output exceeded",
                    ):
                        future.result(timeout=10.0)
                    assert connection.recv(1) == b""


@pytest.mark.parametrize(
    "command",
    [
        (),
        [],
        "python",
        b"python",
        ("",),
        (sys.executable, 1),
        (sys.executable, "nul\0argument"),
    ],
)
def test_general_subprocess_rejects_malformed_command(
    command: Any,
) -> None:
    with pytest.raises(SubprocessError, match="command|executable"):
        run_subprocess(
            command=command,
            input_text="",
            timeout_seconds=1.0,
        )


def test_general_subprocess_preserves_empty_arguments() -> None:
    completed = run_subprocess(
        command=(
            sys.executable,
            "-c",
            "import json, sys; print(json.dumps(sys.argv[1:]))",
            "",
        ),
        input_text="",
        timeout_seconds=2.0,
    )

    assert json.loads(completed.stdout) == [""]


def test_general_subprocess_rejects_non_text_input() -> None:
    with pytest.raises(SubprocessError, match="input must be text"):
        run_subprocess(
            command=(sys.executable, "-c", "pass"),
            input_text=b"bytes",  # type: ignore[arg-type]
            timeout_seconds=1.0,
        )


@pytest.mark.parametrize(
    "environment",
    [
        [],
        {"": "value"},
        {"KEY=OTHER": "value"},
        {"KEY": "nul\0value"},
        {"KEY": 1},
    ],
)
def test_general_subprocess_rejects_malformed_environment(
    environment: Any,
) -> None:
    with pytest.raises(SubprocessError, match="environment"):
        run_subprocess(
            command=(sys.executable, "-c", "pass"),
            input_text="",
            timeout_seconds=1.0,
            environment=environment,
        )


def _recv_exact(channel: socket.socket, size: int) -> bytes:
    received = bytearray()
    while len(received) < size:
        chunk = channel.recv(size - len(received))
        if not chunk:
            break
        received.extend(chunk)
    return bytes(received)


_DEADLOCK_GUARD_SECONDS = 2.0


class _ImmediateExpiryCondition(threading.Condition):
    def __init__(
        self,
        on_expiry: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_expiry = on_expiry
        self._expired = False

    def wait_for(
        self,
        predicate: Callable[[], bool],
        timeout: float | None = None,
    ) -> bool:
        if not self._expired:
            self._expired = True
            assert predicate() is False
            if self._on_expiry is not None:
                self._on_expiry()
            assert predicate() is False
            return False
        return super().wait_for(predicate, timeout)


class _AlwaysExpiredCondition(threading.Condition):
    def __init__(self) -> None:
        super().__init__()
        self.waits: list[float] = []

    def wait_for(
        self,
        predicate: Callable[[], bool],
        timeout: float | None = None,
    ) -> bool:
        assert timeout is not None
        self.waits.append(timeout)
        return predicate()


class _SpuriousThenOverflowCondition(threading.Condition):
    def __init__(self) -> None:
        super().__init__()
        self.trigger: Callable[[], None] | None = None
        self.predicate_checks = 0

    def wait_for(
        self,
        predicate: Callable[[], bool],
        timeout: float | None = None,
    ) -> bool:
        del timeout
        for _ in range(2):
            self.predicate_checks += 1
            assert predicate() is False
        assert self.trigger is not None
        self.trigger()
        self.predicate_checks += 1
        return predicate()


class _ControlledProcess:
    def __init__(self, *, stdout: bytes = b"") -> None:
        self.pid = 12345
        self.returncode: int | None = None
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO()
        self.allow_reap = threading.Event()
        self.reaped = threading.Event()
        self.kill_called = False
        self.wait_calls = 0

    def kill(self) -> None:
        self.kill_called = True
        self.allow_reap.set()

    def wait(self) -> int:
        self.wait_calls += 1
        if not self.allow_reap.wait(_DEADLOCK_GUARD_SECONDS):
            raise RuntimeError("test process reaper was not released")
        self.returncode = 0
        self.reaped.set()
        return self.returncode


class _CleanupProcess:
    def __init__(
        self,
        *,
        returncode: int | None,
        on_kill: Callable[[], None] | None = None,
    ) -> None:
        self.pid = 12345
        self.returncode = returncode
        self.kill_called = False
        self._on_kill = on_kill

    def kill(self) -> None:
        self.kill_called = True
        if self._on_kill is not None:
            self._on_kill()


class _ThreadStub:
    def __init__(
        self,
        *,
        alive: bool = False,
        join_error: Exception | None = None,
    ) -> None:
        self._alive = alive
        self._join_error = join_error
        self.join_timeouts: list[float | None] = []

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)
        if self._join_error is not None:
            raise self._join_error

    def is_alive(self) -> bool:
        return self._alive


def _install_controlled_execution(
    monkeypatch: pytest.MonkeyPatch,
    *,
    process: _ControlledProcess,
    monitor: subprocess_execution._ExecutionMonitor,
) -> None:
    monkeypatch.setattr(
        subprocess_execution._ExecutionMonitor,
        "create",
        staticmethod(lambda: monitor),
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)


def test_execution_monitor_tolerates_spurious_wakeups_and_observes_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    condition = _SpuriousThenOverflowCondition()
    monitor = subprocess_execution._ExecutionMonitor(condition=condition)
    condition.trigger = monitor.record_output_overflow

    def unexpected_observer(*_: object) -> None:
        pytest.fail("an execution trigger must bypass the expiry observer")

    monkeypatch.setattr(os, "waitid", unexpected_observer)

    monitor.wait_for_terminal(12345, 1.0)

    assert condition.predicate_checks == 3
    with pytest.raises(SubprocessOutputLimitError, match="output exceeded"):
        monitor.returncode_or_raise(1.0)


def test_execution_monitor_chunks_above_platform_timeout_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    condition = _AlwaysExpiredCondition()
    monitor = subprocess_execution._ExecutionMonitor(condition=condition)
    timeout_seconds = threading.TIMEOUT_MAX * 2.0 + 1.0
    observed: list[tuple[int, int, int]] = []

    def observe_live(
        identifier_type: int,
        process_id: int,
        options: int,
    ) -> None:
        observed.append((identifier_type, process_id, options))
        return None

    monkeypatch.setattr(os, "waitid", observe_live)

    monitor.wait_for_terminal(12345, timeout_seconds)

    assert all(wait <= threading.TIMEOUT_MAX for wait in condition.waits)
    assert sum(
        (Fraction.from_float(wait) for wait in condition.waits),
        start=Fraction(),
    ) == Fraction.from_float(timeout_seconds)
    assert observed == [
        (
            os.P_PID,
            12345,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
    ]
    with pytest.raises(SubprocessTimeoutError, match="exceeded"):
        monitor.returncode_or_raise(timeout_seconds)


@pytest.mark.parametrize("observer_mode", ["status", "already-reaped"])
def test_apparent_expiry_observes_exit_and_completes_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    observer_mode: str,
) -> None:
    process = _ControlledProcess()

    def release_reaper_before_observer() -> None:
        process.allow_reap.set()
        assert process.reaped.wait(_DEADLOCK_GUARD_SECONDS)

    condition = _ImmediateExpiryCondition(
        release_reaper_before_observer
        if observer_mode == "already-reaped"
        else None
    )
    monitor = subprocess_execution._ExecutionMonitor(condition=condition)
    _install_controlled_execution(
        monkeypatch,
        process=process,
        monitor=monitor,
    )
    observer_calls: list[tuple[int, int, int]] = []

    def observe_exit(
        identifier_type: int,
        process_id: int,
        options: int,
    ) -> object:
        observer_calls.append((identifier_type, process_id, options))
        if observer_mode == "already-reaped":
            assert process.reaped.is_set()
            raise ChildProcessError
        process.allow_reap.set()
        return object()

    def signal_group(_: int) -> None:
        process.allow_reap.set()
        return None

    monkeypatch.setattr(os, "waitid", observe_exit)
    monkeypatch.setattr(
        subprocess_execution,
        "_signal_process_group",
        signal_group,
    )

    completed = run_subprocess(
        command=("controlled",),
        input_text="",
        timeout_seconds=1.0,
    )

    assert completed.returncode == 0
    assert process.wait_calls == 1
    assert monitor.leader_exit_observed is True
    assert monitor.timeout_selected is False
    assert observer_calls == [
        (
            os.P_PID,
            process.pid,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
    ]


def test_live_expiry_selects_timeout_before_late_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _ControlledProcess()
    condition = _ImmediateExpiryCondition()
    monitor = subprocess_execution._ExecutionMonitor(condition=condition)
    _install_controlled_execution(
        monkeypatch,
        process=process,
        monitor=monitor,
    )

    monkeypatch.setattr(os, "waitid", lambda *_args: None)

    def signal_group(_: int) -> None:
        process.allow_reap.set()
        return None

    monkeypatch.setattr(
        subprocess_execution,
        "_signal_process_group",
        signal_group,
    )

    with pytest.raises(SubprocessTimeoutError, match="exceeded 1.0 seconds"):
        run_subprocess(
            command=("controlled",),
            input_text="",
            timeout_seconds=1.0,
        )

    assert process.wait_calls == 1
    assert monitor.timeout_selected is True
    assert monitor.leader_exit_observed is True


def test_observer_error_fails_closed_after_cleanup_and_outranks_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _ControlledProcess()
    condition = _ImmediateExpiryCondition()
    monitor = subprocess_execution._ExecutionMonitor(condition=condition)
    _install_controlled_execution(
        monkeypatch,
        process=process,
        monitor=monitor,
    )
    observer_error = PermissionError(errno.EPERM, "Operation not permitted")

    def fail_observer(*_: object) -> None:
        raise observer_error

    def signal_group(_: int) -> None:
        monitor.record_output_overflow()
        process.allow_reap.set()
        return None

    monkeypatch.setattr(os, "waitid", fail_observer)
    monkeypatch.setattr(
        subprocess_execution,
        "_signal_process_group",
        signal_group,
    )

    with pytest.raises(
        SubprocessInfrastructureError,
        match="execution monitoring failed",
    ) as caught:
        run_subprocess(
            command=("controlled",),
            input_text="",
            timeout_seconds=1.0,
        )

    assert caught.value.__cause__ is observer_error
    assert process.wait_calls == 1
    assert monitor.output_overflow is True


def test_timeout_selection_outranks_late_overflow_and_ipc_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    condition = _AlwaysExpiredCondition()
    monitor = subprocess_execution._ExecutionMonitor(condition=condition)
    monkeypatch.setattr(os, "waitid", lambda *_args: None)

    monitor.wait_for_terminal(12345, 1.0)
    monitor.record_output_overflow()
    monitor.record_io_error(BrokenPipeError("late IPC failure"))
    monitor.record_process_exit(0)

    with pytest.raises(SubprocessTimeoutError, match="exceeded 1.0 seconds"):
        monitor.returncode_or_raise(1.0)


def test_execution_monitor_pins_final_outcome_precedence() -> None:
    monitor = subprocess_execution._ExecutionMonitor.create()
    io_error = OSError("IPC failure")
    monitor_error = OSError("monitor failure")
    monitor.record_process_exit(0)
    monitor.record_io_error(io_error)
    monitor.record_output_overflow()
    with monitor.condition:
        monitor.timeout_selected = True
        monitor.monitor_error = monitor_error

    with pytest.raises(SubprocessInfrastructureError) as caught:
        monitor.returncode_or_raise(1.0)
    assert caught.value.__cause__ is monitor_error

    with monitor.condition:
        monitor.monitor_error = None
    with pytest.raises(SubprocessTimeoutError):
        monitor.returncode_or_raise(1.0)

    with monitor.condition:
        monitor.timeout_selected = False
    with pytest.raises(SubprocessOutputLimitError):
        monitor.returncode_or_raise(1.0)

    with monitor.condition:
        monitor.output_overflow = False
    with pytest.raises(SubprocessInfrastructureError) as caught:
        monitor.returncode_or_raise(1.0)
    assert caught.value.__cause__ is io_error

    with monitor.condition:
        monitor.io_error = None
    assert monitor.returncode_or_raise(1.0) == 0


def test_run_subprocess_has_exactly_one_blocking_process_reaper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _ControlledProcess(stdout=b"captured")
    process.allow_reap.set()
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    signal_attempts: list[int] = []

    def signal_group(process_id: int) -> None:
        signal_attempts.append(process_id)
        return None

    monkeypatch.setattr(
        subprocess_execution,
        "_signal_process_group",
        signal_group,
    )

    completed = run_subprocess(
        command=("controlled",),
        input_text="",
        timeout_seconds=1.0,
    )

    assert completed.stdout == "captured"
    assert process.wait_calls == 1
    assert signal_attempts == [process.pid, process.pid]


@pytest.mark.parametrize(
    "execution_outcome",
    ["timeout", "overflow", "ipc", "normal"],
)
def test_cleanup_failure_outranks_every_execution_outcome(
    monkeypatch: pytest.MonkeyPatch,
    execution_outcome: str,
) -> None:
    process = _CleanupProcess(returncode=0)
    monitor = subprocess_execution._ExecutionMonitor.create()
    monitor.record_process_exit(0)
    if execution_outcome == "timeout":
        with monitor.condition:
            monitor.timeout_selected = True
    elif execution_outcome == "overflow":
        monitor.record_output_overflow()
    elif execution_outcome == "ipc":
        monitor.record_io_error(OSError("IPC failed"))

    def denied_group(*_: object) -> None:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", denied_group)

    with pytest.raises(
        SubprocessInfrastructureError,
        match=r"could not be signaled: errno=1 \(Operation not permitted\)",
    ):
        subprocess_execution._terminate_process_group(  # type: ignore[arg-type]
            process,
            monitor,
            process_waiter=_ThreadStub(),  # type: ignore[arg-type]
            ipc_threads=(),
        )

    assert process.kill_called is False


def test_group_cleanup_uses_direct_kill_only_as_fallback_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = subprocess_execution._ExecutionMonitor.create()
    process = _CleanupProcess(returncode=None)

    def publish_reap() -> None:
        process.returncode = -signal.SIGKILL
        monitor.record_process_exit(-signal.SIGKILL)

    process._on_kill = publish_reap
    group_signal_attempts = 0

    def transient_denial(*_: object) -> None:
        nonlocal group_signal_attempts
        group_signal_attempts += 1
        if group_signal_attempts == 1:
            raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", transient_denial)

    subprocess_execution._terminate_process_group(  # type: ignore[arg-type]
        process,
        monitor,
        process_waiter=_ThreadStub(),  # type: ignore[arg-type]
        ipc_threads=(),
    )

    assert group_signal_attempts == 2
    assert process.kill_called is True
    assert process.returncode == -signal.SIGKILL


def test_reaper_failure_outranks_execution_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = subprocess_execution._ExecutionMonitor.create()
    reaper_error = OSError("wait failed")
    monitor.record_output_overflow()
    monitor.record_reaper_error(reaper_error)
    process = _CleanupProcess(returncode=None)
    monkeypatch.setattr(os, "killpg", lambda *_args: None)

    with pytest.raises(
        SubprocessInfrastructureError,
        match="process group could not be terminated",
    ) as caught:
        subprocess_execution._terminate_process_group(  # type: ignore[arg-type]
            process,
            monitor,
            process_waiter=_ThreadStub(),  # type: ignore[arg-type]
            ipc_threads=(),
        )

    assert caught.value.__cause__ is reaper_error


@pytest.mark.parametrize("failed_thread", ["waiter", "ipc"])
def test_thread_join_failure_outranks_normal_completion(
    monkeypatch: pytest.MonkeyPatch,
    failed_thread: str,
) -> None:
    monitor = subprocess_execution._ExecutionMonitor.create()
    monitor.record_process_exit(0)
    process = _CleanupProcess(returncode=0)
    monkeypatch.setattr(os, "killpg", lambda *_args: None)
    waiter = _ThreadStub(alive=failed_thread == "waiter")
    ipc_threads = (_ThreadStub(alive=True),) if failed_thread == "ipc" else ()

    with pytest.raises(SubprocessInfrastructureError, match="thread"):
        subprocess_execution._terminate_process_group(  # type: ignore[arg-type]
            process,
            monitor,
            process_waiter=waiter,  # type: ignore[arg-type]
            ipc_threads=ipc_threads,  # type: ignore[arg-type]
        )
