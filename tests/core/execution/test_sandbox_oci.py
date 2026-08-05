from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from dr_code.core.execution import sandbox
from dr_code.core.execution.sandbox import (
    MAX_SANDBOX_OUTPUT_BYTES,
    SANDBOX_IMAGE,
    SandboxOutputLimitError,
    SandboxTimeoutError,
    run_python_in_sandbox,
)


# Local runs require an explicit opt-in. CI must run the probes even if its
# workflow-specific opt-in variable drifts or is removed.
pytestmark = [
    pytest.mark.oci,
    pytest.mark.skipif(
        os.environ.get("DR_CODE_RUN_SANDBOX_TESTS") != "1"
        and os.environ.get("CI") is None,
        reason="real OCI sandbox probes require DR_CODE_RUN_SANDBOX_TESTS=1",
    ),
]


@pytest.fixture(scope="module", autouse=True)
def _warm_sandbox_container() -> None:
    # Probe timeouts are watchdogs around asserted terminal outcomes. Pay the
    # runtime cold-start cost once before those deliberately tight bounds.
    run_python_in_sandbox(
        source="input()",
        input_json="{}",
        timeout_seconds=30.0,
    )


def _run(source: str, *, timeout_seconds: float = 2.0) -> None:
    completed = run_python_in_sandbox(
        source=source,
        input_json="{}",
        timeout_seconds=timeout_seconds,
    )
    assert completed.returncode == 0, completed.stderr


def test_provider_and_database_credentials_are_not_inherited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = (
        "ANTHROPIC_API_KEY",
        "DATABASE_URL",
        "DBOS_SYSTEM_DATABASE_URL",
        "OPENAI_API_KEY",
    )
    for name in names:
        monkeypatch.setenv(name, f"operator-secret-{name}")

    _run(
        "import os\n"
        f"names = {names!r}\n"
        "assert all(os.getenv(name) is None for name in names)\n"
    )


def test_operator_file_cannot_be_read(tmp_path: Path) -> None:
    secret_path = tmp_path / "operator-secret"
    secret_path.write_text("do-not-read")

    _run(
        "try:\n"
        f"    open({str(secret_path)!r}).read()\n"
        "except (FileNotFoundError, PermissionError):\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('operator file was readable')\n"
    )


def test_operator_file_cannot_be_written(tmp_path: Path) -> None:
    output_path = tmp_path / "escape"

    _run(
        "try:\n"
        f"    open({str(output_path)!r}, 'w').write('escaped')\n"
        "except (FileNotFoundError, PermissionError):\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('operator file was writable')\n"
    )
    assert output_path.exists() is False


def test_private_ephemeral_working_area_is_writable() -> None:
    _run(
        "with open('/tmp/candidate-file', 'w') as output:\n"
        "    output.write('private')\n"
        "with open('/tmp/candidate-file') as source:\n"
        "    assert source.read() == 'private'\n"
    )


def test_network_connection_is_denied() -> None:
    _run(
        "import socket\n"
        "try:\n"
        "    connection = socket.create_connection(('1.1.1.1', 53), 0.2)\n"
        "except OSError:\n"
        "    pass\n"
        "else:\n"
        "    connection.close()\n"
        "    raise AssertionError('network connection succeeded')\n"
    )


@pytest.mark.parametrize(
    "source",
    [
        (
            "import subprocess\n"
            "try:\n"
            "    subprocess.run(['/bin/true'], check=True)\n"
            "except (OSError, subprocess.SubprocessError):\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('subprocess creation succeeded')\n"
        ),
        (
            "import os\n"
            "try:\n"
            "    os.fork()\n"
            "except OSError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('fork succeeded')\n"
        ),
    ],
    ids=("subprocess", "fork"),
)
def test_additional_processes_are_denied(source: str) -> None:
    _run(source)


def test_timeout_kills_the_launched_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_id = uuid.UUID("bdd4c5e6-df1d-45aa-a084-85753c27fbad")
    container_name = f"dr-code-humaneval-{container_id.hex}"
    monkeypatch.setattr(sandbox.uuid, "uuid4", lambda: container_id)

    with pytest.raises(SandboxTimeoutError):
        run_python_in_sandbox(
            source="while True:\n    pass\n",
            input_json="{}",
            timeout_seconds=0.5,
        )

    runtime = shutil.which("docker")
    assert runtime is not None
    command = [runtime, "container", "inspect", container_name]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"sandbox cleanup inspection timed out: {command!r}")
    assert completed.returncode != 0


def test_stdout_json_ipc_is_bounded() -> None:
    source = (
        "import os\n"
        "input()\n"
        f"os.write(1, b'x' * {MAX_SANDBOX_OUTPUT_BYTES + 1})\n"
    )

    with pytest.raises(SandboxOutputLimitError):
        run_python_in_sandbox(
            source=source,
            input_json="{}",
            timeout_seconds=2.0,
        )


def test_ci_uses_the_documented_immutable_image() -> None:
    assert (
        os.environ.get("DR_CODE_SANDBOX_IMAGE", SANDBOX_IMAGE) == SANDBOX_IMAGE
    )
