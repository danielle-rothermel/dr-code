from __future__ import annotations

import math
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from typing import BinaryIO, Final, Protocol


SANDBOX_IMAGE: Final[str] = (
    "python:3.13.14-slim@"
    "sha256:eb43ff125d8d58d7449dcba7d336c23bcac412f526d861db493b9994d8010280"
)
MAX_SANDBOX_INPUT_BYTES: Final[int] = 1_048_576
MAX_SANDBOX_OUTPUT_BYTES: Final[int] = 1_048_576
SANDBOX_MEMORY_BYTES: Final[int] = 256 * 1024 * 1024
SANDBOX_TMPFS_BYTES: Final[int] = 16 * 1024 * 1024
SANDBOX_OPEN_FILES: Final[int] = 64
# 137 maps resource-limit SIGKILL; 139 maps interpreter-crash SIGSEGV.
CANDIDATE_KILL_RETURNCODES: Final[frozenset[int]] = frozenset({137, 139})
_RUNTIME_ENV: Final[tuple[str, ...]] = (
    "DOCKER_CONFIG",
    "DOCKER_CONTEXT",
    "DOCKER_HOST",
    "HOME",
    "PATH",
    "XDG_RUNTIME_DIR",
)


class SandboxError(RuntimeError):
    pass


class SandboxTimeoutError(SandboxError):
    pass


class SandboxOutputLimitError(SandboxError):
    pass


@dataclass(frozen=True, slots=True)
class SandboxCompletedProcess:
    returncode: int
    stdout: str
    stderr: str


class SandboxRunner(Protocol):
    def __call__(
        self,
        *,
        source: str,
        input_json: str,
        timeout_seconds: float,
    ) -> SandboxCompletedProcess: ...


def run_python_in_sandbox(
    *,
    source: str,
    input_json: str,
    timeout_seconds: float,
) -> SandboxCompletedProcess:
    try:
        payload = input_json.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SandboxError("sandbox input is not valid UTF-8") from exc
    if len(payload) > MAX_SANDBOX_INPUT_BYTES:
        raise SandboxError(
            f"sandbox input exceeded {MAX_SANDBOX_INPUT_BYTES} bytes"
        )
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise SandboxError("sandbox timeout must be finite and positive")

    runtime = _resolve_runtime()
    image = os.environ.get("DR_CODE_SANDBOX_IMAGE", SANDBOX_IMAGE)
    _validate_image_reference(image)
    _require_local_image(runtime, image)

    name = f"dr-code-python-sandbox-{uuid.uuid4().hex}"
    cpu_seconds = max(1, math.ceil(timeout_seconds))
    command = [
        runtime,
        "run",
        "--rm",
        "--interactive",
        "--pull=never",
        "--name",
        name,
        "--label",
        "org.dr-code.python-sandbox=true",
        "--network=none",
        "--read-only",
        "--user=65534:65534",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=1",
        "--cpus=1",
        f"--memory={SANDBOX_MEMORY_BYTES}",
        f"--memory-swap={SANDBOX_MEMORY_BYTES}",
        "--ulimit",
        f"cpu={cpu_seconds}:{cpu_seconds}",
        "--ulimit",
        f"fsize={MAX_SANDBOX_OUTPUT_BYTES}:{MAX_SANDBOX_OUTPUT_BYTES}",
        "--ulimit",
        f"nofile={SANDBOX_OPEN_FILES}:{SANDBOX_OPEN_FILES}",
        "--tmpfs",
        (
            "/tmp:rw,noexec,nosuid,nodev,uid=65534,gid=65534,mode=700,size="
            f"{SANDBOX_TMPFS_BYTES}"
        ),
        "--workdir=/tmp",
        "--env=HOME=/tmp",
        "--env=PYTHONDONTWRITEBYTECODE=1",
        "--env=PYTHONHASHSEED=0",
        image,
        "python",
        "-I",
        "-c",
        source,
    ]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=_runtime_environment(),
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise SandboxError("sandbox process failed to start") from exc
    stdout = bytearray()
    stderr = bytearray()
    overflow = threading.Event()
    io_failed = threading.Event()
    io_errors: list[BaseException] = []
    threads = [
        threading.Thread(
            target=_write_input,
            args=(process.stdin, payload, io_failed, io_errors),
            daemon=True,
        ),
        threading.Thread(
            target=_read_bounded,
            args=(
                process.stdout,
                stdout,
                overflow,
                io_failed,
                io_errors,
            ),
            daemon=True,
        ),
        threading.Thread(
            target=_read_bounded,
            args=(
                process.stderr,
                stderr,
                overflow,
                io_failed,
                io_errors,
            ),
            daemon=True,
        ),
    ]
    deadline = time.monotonic() + timeout_seconds
    execution_error: SandboxError | None = None
    termination_error: SandboxError | None = None
    started_threads: list[threading.Thread] = []
    try:
        for thread in threads:
            try:
                thread.start()
            except RuntimeError as exc:
                execution_error = SandboxError(
                    "sandbox IPC threads failed to start"
                )
                execution_error.__cause__ = exc
                break
            started_threads.append(thread)

        if execution_error is None:
            while process.poll() is None:
                if overflow.is_set():
                    execution_error = SandboxOutputLimitError(
                        "sandbox output exceeded "
                        f"{MAX_SANDBOX_OUTPUT_BYTES} bytes"
                    )
                    break
                if io_failed.is_set():
                    execution_error = SandboxError("sandbox IPC failed")
                    execution_error.__cause__ = io_errors[0]
                    break
                if time.monotonic() >= deadline:
                    execution_error = SandboxTimeoutError(
                        f"sandbox exceeded {timeout_seconds} seconds"
                    )
                    break
                time.sleep(0.01)
    finally:
        if process.poll() is None:
            try:
                _terminate_container(runtime, name, process)
            except SandboxError as exc:
                termination_error = exc
        for thread in started_threads:
            thread.join(timeout=1.0)

    if termination_error is not None:
        if execution_error is not None:
            cleanup_cause = termination_error.__cause__
            if execution_error.__cause__ is None and cleanup_cause is not None:
                execution_error.__cause__ = cleanup_cause
            raise termination_error from execution_error
        raise termination_error
    if any(thread.is_alive() for thread in started_threads):
        raise SandboxError("sandbox IPC threads could not be terminated")
    if execution_error is not None:
        raise execution_error
    if overflow.is_set():
        raise SandboxOutputLimitError(
            f"sandbox output exceeded {MAX_SANDBOX_OUTPUT_BYTES} bytes"
        )
    if io_failed.is_set():
        raise SandboxError("sandbox IPC failed") from io_errors[0]
    return SandboxCompletedProcess(
        returncode=process.returncode,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
    )


def _resolve_runtime() -> str:
    runtime = shutil.which("docker")
    if runtime is None:
        raise SandboxError("sandbox runtime is unavailable: docker")
    return runtime


def _validate_image_reference(image: str) -> None:
    name, separator, digest = image.rpartition("@")
    if (
        not name
        or any(character.isspace() for character in name)
        or separator != "@"
        or not digest.startswith("sha256:")
    ):
        raise SandboxError(
            "DR_CODE_SANDBOX_IMAGE must use an immutable sha256 digest"
        )
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise SandboxError(
            "DR_CODE_SANDBOX_IMAGE must use an immutable sha256 digest"
        )


def _runtime_environment() -> dict[str, str]:
    return {
        name: os.environ[name] for name in _RUNTIME_ENV if name in os.environ
    }


def _require_local_image(runtime: str, image: str) -> None:
    try:
        completed = subprocess.run(
            [runtime, "image", "inspect", image],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
            env=_runtime_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SandboxError(
            f"sandbox image inspection failed: {image}"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise SandboxError(
            f"sandbox image is not available locally: {image}; {detail}"
        )


def _write_input(
    stream: BinaryIO | None,
    payload: bytes,
    failed: threading.Event,
    errors: list[BaseException],
) -> None:
    if stream is None:
        return
    try:
        stream.write(payload)
        stream.write(b"\n")
        stream.close()
    except BrokenPipeError:
        return
    except BaseException as exc:
        errors.append(exc)
        failed.set()


def _read_bounded(
    stream: BinaryIO | None,
    output: bytearray,
    overflow: threading.Event,
    failed: threading.Event,
    errors: list[BaseException],
) -> None:
    if stream is None:
        return
    try:
        while chunk := stream.read(65_536):
            remaining = MAX_SANDBOX_OUTPUT_BYTES - len(output)
            output.extend(chunk[:remaining])
            if len(chunk) > remaining:
                overflow.set()
                return
    except BaseException as exc:
        errors.append(exc)
        failed.set()


def _terminate_container(
    runtime: str,
    name: str,
    process: subprocess.Popen[bytes],
) -> None:
    environment = _runtime_environment()
    process_group_error: OSError | None = None
    for command in (
        [runtime, "kill", "--signal=KILL", name],
        [runtime, "rm", "--force", name],
    ):
        try:
            subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
                env=environment,
            )
        except (OSError, subprocess.SubprocessError):
            continue
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            process_group_error = exc
    try:
        process.wait(timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SandboxError(
            "sandbox process group could not be terminated"
        ) from exc

    inspected = _cleanup_command(
        [runtime, "container", "inspect", name],
        environment=environment,
    )
    if inspected.returncode == 0:
        raise SandboxError("sandbox container survived forced termination")
    runtime_status = _cleanup_command(
        [runtime, "info"],
        environment=environment,
    )
    if runtime_status.returncode != 0:
        raise SandboxError(
            "sandbox runtime could not confirm container termination"
        )
    if process_group_error is not None:
        raise SandboxError(
            "sandbox process group could not be terminated"
        ) from process_group_error


def _cleanup_command(
    command: list[str],
    *,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SandboxError("sandbox cleanup command failed") from exc
