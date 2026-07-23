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

from dr_code.humaneval import subprocess_runner
from dr_code.humaneval.subprocess_runner import (
    MAX_SUBPROCESS_INPUT_BYTES,
    MAX_SUBPROCESS_OUTPUT_BYTES,
    SubprocessError,
    SubprocessOutputLimitError,
    SubprocessTimeoutError,
    run_python_subprocess,
)


def test_subprocess_round_trips_json_in_a_fresh_python_child() -> None:
    first = run_python_subprocess(
        source=(
            "import json, os, sys\n"
            "payload = json.load(sys.stdin)\n"
            "print(json.dumps({'payload': payload, 'pid': os.getpid()}))\n"
        ),
        input_json='{"value": 7}',
        timeout_seconds=2.0,
    )
    second = run_python_subprocess(
        source="import json, os; print(json.dumps({'pid': os.getpid()}))",
        input_json="{}",
        timeout_seconds=2.0,
    )

    first_output = json.loads(first.stdout)
    second_output = json.loads(second.stdout)
    assert first.returncode == 0
    assert first.stderr == ""
    assert first_output["payload"] == {"value": 7}
    assert first_output["pid"] != os.getpid()
    assert first_output["pid"] != second_output["pid"]


def test_subprocess_uses_minimal_environment_with_one_openblas_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DR_CODE_SHOULD_NOT_REACH_CHILD", "secret")

    completed = run_python_subprocess(
        source="import json, os; print(json.dumps(dict(os.environ)))",
        input_json="{}",
        timeout_seconds=2.0,
    )

    environment = json.loads(completed.stdout)
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
    assert "DR_CODE_SHOULD_NOT_REACH_CHILD" not in environment


def test_subprocess_invokes_sys_executable_without_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fail_start(command: list[str], **kwargs: Any) -> None:
        seen["command"] = command
        seen.update(kwargs)
        raise OSError("stop after inspection")

    monkeypatch.setattr(subprocess, "Popen", fail_start)

    with pytest.raises(SubprocessError, match="could not start"):
        run_python_subprocess(
            source="pass",
            input_json="{}",
            timeout_seconds=1.0,
        )

    assert seen["command"] == [sys.executable, "-I", "-c", "pass"]
    assert seen["start_new_session"] is True
    assert seen["env"] == {"OPENBLAS_NUM_THREADS": "1"}
    assert "docker" not in seen["command"]


@pytest.mark.parametrize("timeout_seconds", [0.0, -1.0, math.inf, math.nan])
def test_subprocess_rejects_invalid_timeout(timeout_seconds: float) -> None:
    with pytest.raises(
        SubprocessError,
        match="timeout must be finite and positive",
    ):
        run_python_subprocess(
            source="pass",
            input_json="{}",
            timeout_seconds=timeout_seconds,
        )


def test_subprocess_rejects_oversized_input_before_starting_child() -> None:
    with pytest.raises(SubprocessError, match="input exceeded"):
        run_python_subprocess(
            source="pass",
            input_json="x" * (MAX_SUBPROCESS_INPUT_BYTES + 1),
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
            input_json="{}",
            timeout_seconds=2.0,
        )


def test_subprocess_output_limit_is_shared_by_stdout_and_stderr() -> None:
    half_limit = MAX_SUBPROCESS_OUTPUT_BYTES // 2 + 1

    with pytest.raises(SubprocessOutputLimitError, match="output exceeded"):
        run_python_subprocess(
            source=(
                "import os\n"
                f"os.write(1, b'x' * {half_limit})\n"
                f"os.write(2, b'y' * {half_limit})\n"
            ),
            input_json="{}",
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
        run_python_subprocess(
            source=source,
            input_json="{}",
            timeout_seconds=0.2,
        )

    descendant_pid = int(pid_path.read_text())
    deadline = time.monotonic() + 2.0
    while _process_exists(descendant_pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not _process_exists(descendant_pid)


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


def test_candidate_signal_returncode_uses_host_subprocess_semantics() -> None:
    completed = run_python_subprocess(
        source=(
            "import os, signal\n"
            "os.kill(os.getpid(), signal.SIGKILL)\n"
        ),
        input_json="{}",
        timeout_seconds=2.0,
    )

    assert completed.returncode == -signal.SIGKILL


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
    group_signal_attempted = False

    def stale_group(*_: object) -> None:
        nonlocal group_signal_attempted
        group_signal_attempted = True
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", stale_group)

    subprocess_runner._terminate_process_group(process)  # type: ignore[arg-type]

    assert group_signal_attempted is True
    assert process.kill_called is False
    assert process.wait_called is True


def test_live_process_group_signal_error_remains_infrastructure_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _ProcessStub(returncode=None)

    def denied_group(*_: object) -> None:
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", denied_group)

    with pytest.raises(
        SubprocessError,
        match=r"could not be signaled: errno=1 \(Operation not permitted\)",
    ):
        subprocess_runner._terminate_process_group(  # type: ignore[arg-type]
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

    subprocess_runner._terminate_process_group(process)  # type: ignore[arg-type]

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

    subprocess_runner._terminate_process_group(process)  # type: ignore[arg-type]

    assert group_signal_attempts == 2
    assert process.kill_called is True
    assert process.wait_called is True
    assert process.returncode == -signal.SIGKILL
