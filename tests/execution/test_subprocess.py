from __future__ import annotations

import errno
import json
import math
import os
import signal
import subprocess
import sys
import time
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


def test_subprocess_timeout_kills_descendant_process_group(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "descendant.pid"
    source = (
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid))\n"
        "while True: time.sleep(1)\n"
    )

    with pytest.raises(SubprocessTimeoutError, match="exceeded 0.2 seconds"):
        run_subprocess(
            command=(sys.executable, "-c", source),
            input_text="",
            timeout_seconds=0.2,
        )

    descendant_pid = int(pid_path.read_text())
    deadline = time.monotonic() + 2.0
    while _process_exists(descendant_pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not _process_exists(descendant_pid)


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


def _process_exists(pid: int) -> bool:
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        try:
            if proc_stat.read_text().split()[2] == "Z":
                return False
        except (OSError, IndexError):
            pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class _ProcessStub:
    def __init__(self, *, returncode: int | None) -> None:
        self.pid = 12345
        self.returncode = returncode
        self.kill_called = False
        self.wait_called = False

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.kill_called = True
        self.returncode = -signal.SIGKILL

    def wait(self, timeout: float) -> int:
        self.wait_called = True
        assert timeout > 0
        assert self.returncode is not None
        return self.returncode


def test_post_reap_group_signal_error_is_not_infrastructure_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _ProcessStub(returncode=0)

    def stale_group(*_: object) -> None:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", stale_group)

    subprocess_execution._terminate_process_group(  # type: ignore[arg-type]
        process
    )

    assert process.kill_called is False
    assert process.wait_called is True


def test_live_group_signal_error_is_infrastructure_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _ProcessStub(returncode=None)

    def denied_group(*_: object) -> None:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", denied_group)

    with pytest.raises(
        SubprocessInfrastructureError,
        match=r"could not be signaled: errno=1 \(Operation not permitted\)",
    ):
        subprocess_execution._terminate_process_group(  # type: ignore[arg-type]
            process
        )

    assert process.kill_called is True
    assert process.wait_called is True


def test_completion_during_direct_kill_clears_stale_group_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _ProcessStub(returncode=None)

    def denied_group(*_: object) -> None:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    def completion_wins_kill_race() -> None:
        process.kill_called = True
        process.returncode = 0

    monkeypatch.setattr(os, "killpg", denied_group)
    monkeypatch.setattr(process, "kill", completion_wins_kill_race)

    subprocess_execution._terminate_process_group(  # type: ignore[arg-type]
        process
    )

    assert process.kill_called is True
    assert process.wait_called is True
    assert process.returncode == 0


def test_group_cleanup_retry_preserves_descendant_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _ProcessStub(returncode=None)
    group_signal_attempts = 0

    def transient_denial(*_: object) -> None:
        nonlocal group_signal_attempts
        group_signal_attempts += 1
        if group_signal_attempts == 1:
            raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", transient_denial)

    subprocess_execution._terminate_process_group(  # type: ignore[arg-type]
        process
    )

    assert group_signal_attempts == 2
    assert process.kill_called is True
    assert process.wait_called is True
    assert process.returncode == -signal.SIGKILL
