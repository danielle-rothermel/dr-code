from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from dr_code.humaneval.sandbox import (
    MAX_SANDBOX_OUTPUT_BYTES,
    SANDBOX_IMAGE,
    SandboxOutputLimitError,
    run_python_in_sandbox,
)
from dr_code.humaneval.batch_runner import evaluate_human_eval_code
from dr_code.humaneval.scoring import CompletedScore, score_humaneval_submission
from dr_code.humaneval.task import HumanEvalTask


# The probes are opt-in locally but always run in CI: they must fail loudly
# there (missing runtime/image) rather than skip if workflow env wiring drifts.
pytestmark = pytest.mark.skipif(
    os.environ.get("DR_CODE_RUN_SANDBOX_TESTS") != "1"
    and os.environ.get("CI") is None,
    reason="real OCI sandbox probes require DR_CODE_RUN_SANDBOX_TESTS=1",
)


@pytest.fixture(scope="module", autouse=True)
def _warm_sandbox_container() -> None:
    # The first probe to run on a fresh CI runner pays container cold-start,
    # which can push it past a tight per-probe deadline (e.g. the 2.0s
    # `test_known_good_submission_scores_inside_real_sandbox`) and flake as a
    # spurious timeout. Warm the runtime once here with a generous timeout so
    # the timed probes measure steady-state execution, not image/container
    # startup. Probe timeouts stay unchanged so their timing assertions remain
    # meaningful.
    run_python_in_sandbox(
        source="input()\nprint('[]')\n",
        input_json="{}",
        timeout_seconds=30.0,
    )


def _task() -> HumanEvalTask:
    return HumanEvalTask(
        task_id="HumanEval/sandbox-fixture",
        prompt="def add_one(x):\n",
        canonical_solution="    return x + 1\n",
        entry_point="add_one",
        test=(
            "def check(candidate):\n"
            "    inputs = [(1,), (2,)]\n"
            "    results = [2, 3]\n"
            "    for inp, expected in zip(inputs, results):\n"
            "        assertion(candidate(*inp), expected)\n"
        ),
    )


def _evaluate(candidate_code: str, *, timeout: float = 2.0) -> bool:
    return evaluate_human_eval_code(
        task=_task(),
        candidate_code=candidate_code,
        timeout_seconds=timeout,
    ).passed


def test_known_good_submission_scores_inside_real_sandbox() -> None:
    result = score_humaneval_submission(
        raw_submission="def add_one(x):\n    return x + 1\n",
        task=_task(),
        timeout_seconds=2.0,
    )

    assert isinstance(result, CompletedScore)
    assert result.score == 1.0


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
    candidate = (
        "import os\n"
        "def add_one(x):\n"
        f"    names = {names!r}\n"
        "    return x + 1 if all(os.getenv(name) is None for name in names) "
        "else -1\n"
    )

    assert _evaluate(candidate) is True


def test_operator_file_cannot_be_read(
    tmp_path: Path,
) -> None:
    secret_path = tmp_path / "operator-secret"
    secret_path.write_text("do-not-read")
    candidate = (
        "def add_one(x):\n"
        "    try:\n"
        f"        open({str(secret_path)!r}).read()\n"
        "    except (FileNotFoundError, PermissionError):\n"
        "        return x + 1\n"
        "    return -1\n"
    )

    assert _evaluate(candidate) is True


def test_operator_file_cannot_be_written(tmp_path: Path) -> None:
    output_path = tmp_path / "escape"
    candidate = (
        "def add_one(x):\n"
        "    try:\n"
        f"        open({str(output_path)!r}, 'w').write('escaped')\n"
        "    except (FileNotFoundError, PermissionError):\n"
        "        return x + 1\n"
        "    return -1\n"
    )

    assert _evaluate(candidate) is True
    assert output_path.exists() is False


def test_private_ephemeral_working_area_is_writable() -> None:
    candidate = (
        "def add_one(x):\n"
        "    with open('/tmp/candidate-file', 'w') as output:\n"
        "        output.write('private')\n"
        "    with open('/tmp/candidate-file') as source:\n"
        "        return x + 1 if source.read() == 'private' else -1\n"
    )

    assert _evaluate(candidate) is True


def test_network_connection_is_denied() -> None:
    candidate = (
        "import socket\n"
        "def add_one(x):\n"
        "    try:\n"
        "        connection = socket.create_connection(('1.1.1.1', 53), 0.2)\n"
        "    except OSError:\n"
        "        return x + 1\n"
        "    connection.close()\n"
        "    return -1\n"
    )

    assert _evaluate(candidate) is True


@pytest.mark.parametrize(
    "candidate",
    [
        (
            "import subprocess\n"
            "def add_one(x):\n"
            "    try:\n"
            "        subprocess.run(['/bin/true'], check=True)\n"
            "    except (OSError, subprocess.SubprocessError):\n"
            "        return x + 1\n"
            "    return -1\n"
        ),
        (
            "import os\n"
            "def add_one(x):\n"
            "    try:\n"
            "        os.fork()\n"
            "    except OSError:\n"
            "        return x + 1\n"
            "    return -1\n"
        ),
    ],
    ids=("subprocess", "fork"),
)
def test_additional_processes_are_denied(candidate: str) -> None:
    assert _evaluate(candidate) is True


def test_timeout_kills_the_complete_container() -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=(
            "import os\n"
            "def add_one(x):\n"
            "    try:\n"
            "        os.fork()\n"
            "    except OSError:\n"
            "        pass\n"
            "    while True:\n"
            "        pass\n"
        ),
        timeout_seconds=0.5,
    )

    assert result.status_counts == {"timeout": 2}
    runtime = shutil.which(os.environ.get("DR_CODE_SANDBOX_RUNTIME", "docker"))
    assert runtime is not None
    completed = subprocess.run(
        [
            runtime,
            "ps",
            "--quiet",
            "--filter",
            "label=org.dr-code.humaneval-sandbox=true",
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    assert completed.stdout.strip() == ""


def test_candidate_sys_exit_is_scored_not_harness_failure() -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=(
            "import sys\n"
            "sys.exit(0)\n"
            "def add_one(x):\n"
            "    return x + 1\n"
        ),
        timeout_seconds=2.0,
    )

    assert result.passed is False
    assert result.status_counts == {"error": 2}


def test_memory_exhaustion_is_scored_not_harness_failure() -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=(
            "def add_one(x):\n"
            "    data = bytearray(512 * 1024 * 1024)\n"
            "    return x + 1 + data[0]\n"
        ),
        timeout_seconds=5.0,
    )

    assert result.passed is False
    assert set(result.status_counts) <= {"error", "timeout"}


def test_output_flood_is_scored_not_harness_failure() -> None:
    result = evaluate_human_eval_code(
        task=_task(),
        candidate_code=(
            "import sys\n"
            "def add_one(x):\n"
            f"    sys.stdout.write('x' * {MAX_SANDBOX_OUTPUT_BYTES + 1})\n"
            "    return x + 1\n"
        ),
        timeout_seconds=5.0,
    )

    assert result.passed is False
    assert result.status_counts == {"error": 2}


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
    assert os.environ.get("DR_CODE_SANDBOX_IMAGE", SANDBOX_IMAGE) == SANDBOX_IMAGE
