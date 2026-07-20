"""OCI isolation boundary for executing model-generated Python."""

from __future__ import annotations

import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from typing import BinaryIO, Final, Protocol


SANDBOX_IMAGE: Final[str] = (
    "python:3.13.14-slim@"
    "sha256:eb43ff125d8d58d7449dcba7d336c23bcac412f526d861db493b9994d8010280"
)
# HumanEval+/113 alone serializes to 1,255,579 bytes.  Four MiB admits that
# payload and up to roughly 0.5 MiB of escaped candidate source while keeping
# the host-to-container IPC input bounded.  The output bound stays tighter.
MAX_SANDBOX_INPUT_BYTES: Final[int] = 4 * 1024 * 1024
MAX_SANDBOX_OUTPUT_BYTES: Final[int] = 1_048_576
SANDBOX_MEMORY_BYTES: Final[int] = 256 * 1024 * 1024
SANDBOX_TMPFS_BYTES: Final[int] = 16 * 1024 * 1024
SANDBOX_OPEN_FILES: Final[int] = 64
SANDBOX_STARTUP_TIMEOUT_SECONDS: Final[float] = 10.0
# Container exit codes attributable to the candidate hitting a sandbox
# resource boundary rather than to a broken sandbox: 137 is SIGKILL from the
# memory limit or the CPU hard limit (python runs as container PID 1, which
# ignores SIGXCPU, so the kernel escalates to SIGKILL), and 139 is SIGSEGV
# from an interpreter crash.
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
    """The trusted sandbox boundary could not safely complete execution."""


class SandboxTimeoutError(SandboxError):
    """The sandbox exceeded its wall-clock deadline."""


class SandboxOutputLimitError(SandboxError):
    """The sandbox emitted more data than the bounded IPC contract permits."""


@dataclass(frozen=True, slots=True)
class SandboxCompletedProcess:
    returncode: int
    stdout: str
    stderr: str


class SandboxRunner(Protocol):
    """The callable contract for executing candidate code in isolation.

    `run_python_in_sandbox` is the production implementation; the evaluation
    entry points accept any conforming callable so tests can substitute a
    local runner without patching module globals.
    """

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
    """Run Python in a locked-down OCI container using bounded JSON IPC.

    The runtime and image are trusted deployment dependencies. The image must
    already exist locally: scored code can never trigger a registry pull.
    """
    payload = input_json.encode("utf-8")
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

    name = f"dr-code-humaneval-{uuid.uuid4().hex}"
    cidfile = Path(gettempdir()) / f"{name}.cid"
    cpu_seconds = max(1, math.ceil(timeout_seconds))
    command = [
        runtime,
        "run",
        "--rm",
        "--interactive",
        "--pull=never",
        "--cidfile",
        str(cidfile),
        "--name",
        name,
        "--label",
        "org.dr-code.humaneval-sandbox=true",
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
        # NumPy's bundled OpenBLAS otherwise creates worker threads during
        # import, which conflicts with the deliberate one-process limit.
        "--env=OPENBLAS_NUM_THREADS=1",
        image,
        "python",
        "-I",
        "-c",
        source,
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env=_runtime_environment(),
    )
    stdout = bytearray()
    stderr = bytearray()
    overflow = threading.Event()
    io_errors: list[BaseException] = []
    threads = [
        threading.Thread(
            target=_write_input,
            args=(process.stdin, payload, io_errors),
            daemon=True,
        ),
        threading.Thread(
            target=_read_bounded,
            args=(process.stdout, stdout, overflow, io_errors),
            daemon=True,
        ),
        threading.Thread(
            target=_read_bounded,
            args=(process.stderr, stderr, overflow, io_errors),
            daemon=True,
        ),
    ]
    started_threads: list[threading.Thread] = []
    container_id: str | None = None
    startup_timed_out = False
    timed_out = False
    try:
        for thread in threads:
            thread.start()
            started_threads.append(thread)

        startup_deadline = time.monotonic() + SANDBOX_STARTUP_TIMEOUT_SECONDS
        while (
            container_id is None
            and process.poll() is None
            and time.monotonic() < startup_deadline
        ):
            container_id = _read_container_id(cidfile)
            if container_id is not None:
                break
            time.sleep(0.01)
        if container_id is None:
            container_id = _read_container_id(cidfile)
        startup_timed_out = container_id is None and process.poll() is None
        deadline = (
            time.monotonic()
            if startup_timed_out
            else time.monotonic() + timeout_seconds
        )
        while process.poll() is None:
            if overflow.is_set():
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(0.01)
    finally:
        active_exception = sys.exception()
        try:
            returncode = process.poll()
            if returncode is None or container_id is None:
                try:
                    _terminate_container(
                        runtime, container_id or name, process
                    )
                except SandboxError as exc:
                    if active_exception is not None:
                        raise exc from active_exception
                    raise
        finally:
            for thread in started_threads:
                thread.join(timeout=1.0)
            cidfile.unlink(missing_ok=True)

    if any(thread.is_alive() for thread in threads):
        raise SandboxError("sandbox IPC threads could not be terminated")
    if startup_timed_out:
        raise SandboxError(
            "sandbox container did not start within "
            f"{SANDBOX_STARTUP_TIMEOUT_SECONDS} seconds"
        )
    if container_id is None:
        raise SandboxError(
            "sandbox runtime exited before producing a container ID"
        )
    if timed_out:
        raise SandboxTimeoutError(
            f"sandbox exceeded {timeout_seconds} seconds"
        )
    if overflow.is_set():
        raise SandboxOutputLimitError(
            f"sandbox output exceeded {MAX_SANDBOX_OUTPUT_BYTES} bytes"
        )
    if io_errors:
        raise SandboxError("sandbox IPC failed") from io_errors[0]
    return SandboxCompletedProcess(
        returncode=process.returncode,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
    )


def _resolve_runtime() -> str:
    configured = os.environ.get("DR_CODE_SANDBOX_RUNTIME", "docker")
    if configured not in {"docker", "podman"}:
        raise SandboxError(
            "DR_CODE_SANDBOX_RUNTIME must be 'docker' or 'podman'"
        )
    runtime = shutil.which(configured)
    if runtime is None:
        raise SandboxError(f"sandbox runtime is unavailable: {configured}")
    return runtime


def _read_container_id(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else None


def _validate_image_reference(image: str) -> None:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image) is not None:
        return

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


def _read_bounded(
    stream: BinaryIO | None,
    output: bytearray,
    overflow: threading.Event,
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


def _terminate_container(
    runtime: str,
    name: str,
    process: subprocess.Popen[bytes],
) -> None:
    environment = _runtime_environment()
    try:
        subprocess.run(
            [runtime, "kill", "--signal=KILL", name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError):
        pass
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            # On macOS the runtime CLI's process group can include a helper
            # that the caller cannot signal, even though the direct child is
            # still ours.  Kill that child directly; the container itself was
            # already addressed by immutable name above and is verified gone
            # below.
            try:
                process.kill()
            except OSError as exc:
                raise SandboxError(
                    "sandbox process could not be signaled"
                ) from exc
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired as exc:
        raise SandboxError(
            "sandbox process group could not be terminated"
        ) from exc

    # Only remove after the `run` client has exited. Removing while that
    # client is still completing its create/start request can race with the
    # daemon and leave a late-created container behind.
    removal_deadline = time.monotonic() + 5.0
    while True:
        _cleanup_command(
            [runtime, "rm", "--force", name], environment=environment
        )
        inspected = _cleanup_command(
            [runtime, "container", "inspect", name],
            environment=environment,
        )
        if inspected.returncode != 0:
            break
        if time.monotonic() >= removal_deadline:
            raise SandboxError("sandbox container survived forced termination")
        time.sleep(0.05)
    runtime_status = _cleanup_command(
        [runtime, "info"],
        environment=environment,
    )
    if runtime_status.returncode != 0:
        raise SandboxError(
            "sandbox runtime could not confirm container termination"
        )


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
