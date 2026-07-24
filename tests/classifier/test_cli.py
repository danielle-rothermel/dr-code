from __future__ import annotations

import json
import os
import select
import signal
import socket
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from dr_code.classifier import cli as classifier_cli
from dr_code.classifier.lane import LanePolicy
from dr_code.viewer.cli import app
from dr_code.viewer.database import (
    DatabaseOwnershipError,
    database_owner_lock_path,
)
from viewer.helpers import write_bundle


class FixedLane:
    provider = "provider"
    model = "model"
    policy = LanePolicy(adapter="test-fixed-lane-v1")

    def complete(self, prompt: str) -> str:
        label = (
            "wrong-algorithm"
            if "Classify one test failure." in prompt
            else "prose-no-code"
        )
        return json.dumps({"label": label, "rationale": "fixed"})


def _descriptor_file(
    tmp_path: Path,
    *,
    run_id: str = "fixture-run",
) -> Path:
    descriptor = write_bundle(
        tmp_path / "bundle",
        run_id=run_id,
        dataset_id="org/data",
        task_namespace="Task",
    )
    assert descriptor.evaluation_root_path is not None
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "label": descriptor.label,
                "dataset_id": descriptor.dataset_id,
                "corpus": str(descriptor.corpus_path),
                "preprocessing": str(
                    descriptor.preprocessing_manifest_path.parent
                ),
                "candidate_evaluation": str(descriptor.evaluation_root_path),
            }
        )
    )
    return path


def test_cli_accepts_descriptor_directly_and_emits_sorted_json(
    tmp_path, monkeypatch
) -> None:
    path = _descriptor_file(tmp_path)
    monkeypatch.setattr(
        classifier_cli,
        "SubscriptionLane",
        lambda **kwargs: FixedLane(),
    )
    result = CliRunner().invoke(
        app,
        [
            "classify-failures",
            str(path),
            "--provider",
            "provider",
            "--model",
            "model",
            "--database",
            str(tmp_path / "state.duckdb"),
            "--repeats",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dataset_id"] == "org/data"
    assert payload["classified"] == 6
    assert result.output.strip() == json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert "failure-classifications" in payload["details_path"]
    filename = Path(payload["details_path"]).stem
    assert len(filename) == 64
    assert filename == payload["experiment_identity"]


def test_default_paths_are_bounded_for_arbitrarily_long_run_ids(
    tmp_path,
    monkeypatch,
) -> None:
    path = _descriptor_file(tmp_path, run_id="run-" + "x" * 10_000)
    monkeypatch.setattr(
        classifier_cli,
        "SubscriptionLane",
        lambda **kwargs: FixedLane(),
    )

    result = CliRunner().invoke(
        app,
        [
            "classify-failures",
            str(path),
            "--provider",
            "provider",
            "--model",
            "model",
            "--database",
            str(tmp_path / "state.duckdb"),
            "--repeats",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    details = Path(json.loads(result.output)["details_path"])
    assert len(details.name) == 70
    assert len(classifier_cli._staged_artifact_path(details).name) < 100
    assert len(classifier_cli._output_lock_path(details).name) < 100


def test_database_ownership_error_is_a_clean_cli_parameter_failure(
    tmp_path,
    monkeypatch,
) -> None:
    path = _descriptor_file(tmp_path)
    monkeypatch.setattr(
        classifier_cli,
        "SubscriptionLane",
        lambda **kwargs: FixedLane(),
    )

    class OwnedDatabase:
        def __init__(self, path: Path) -> None:
            raise DatabaseOwnershipError(f"viewer database is in use: {path}")

    monkeypatch.setattr(classifier_cli, "ViewerDatabase", OwnedDatabase)
    result = CliRunner().invoke(
        app,
        [
            "classify-failures",
            str(path),
            "--provider",
            "provider",
            "--model",
            "model",
            "--database",
            str(tmp_path / "state.duckdb"),
            "--repeats",
            "1",
        ],
    )

    assert result.exit_code == 2
    assert "viewer database is in use" in result.output
    failure_directory = tmp_path / "failure-classifications"
    assert not tuple(failure_directory.glob("*.jsonl"))
    assert not tuple(failure_directory.glob("*.publication"))


def test_cli_has_explicit_provider_model_timeout_options() -> None:
    command = get_command(app).commands["classify-failures"]
    options = {
        parameter.name: tuple(parameter.opts) for parameter in command.params
    }

    assert options["provider"] == ("--provider",)
    assert options["model"] == ("--model",)
    assert options["timeout"] == ("--timeout",)


@pytest.mark.parametrize("database_is_symlink", [False, True])
def test_cli_rejects_database_equal_to_stage_before_open_or_mutation(
    tmp_path,
    monkeypatch,
    database_is_symlink,
) -> None:
    descriptor_path = _descriptor_file(tmp_path)
    details_path = tmp_path / "details.jsonl"
    stage_path = tmp_path / ".details.jsonl.publication"
    lock_path = tmp_path / ".details.jsonl.lock"
    database_path = stage_path
    if database_is_symlink:
        database_path = tmp_path / "database-alias"
        database_path.symlink_to(stage_path)
    monkeypatch.setattr(
        classifier_cli,
        "SubscriptionLane",
        lambda **kwargs: FixedLane(),
    )

    class DatabaseMustNotOpen:
        def __init__(self, path: Path) -> None:
            raise AssertionError(f"DuckDB opened at {path}")

    monkeypatch.setattr(classifier_cli, "ViewerDatabase", DatabaseMustNotOpen)
    result = CliRunner().invoke(
        app,
        [
            "classify-failures",
            str(descriptor_path),
            "--provider",
            "provider",
            "--model",
            "model",
            "--database",
            str(database_path),
            "--details",
            str(details_path),
            "--repeats",
            "1",
        ],
    )

    assert result.exit_code == 2
    assert "database path collides with classification" in result.output
    assert "staged artifact path" in result.output
    assert not stage_path.exists()
    assert not lock_path.exists()


def test_cli_rejects_details_equal_to_database_owner_lock_before_open(
    tmp_path,
    monkeypatch,
) -> None:
    descriptor_path = _descriptor_file(tmp_path)
    database_path = tmp_path / "state.duckdb"
    details_path = database_owner_lock_path(database_path)
    descriptor = classifier_cli.RunDescriptor.from_file(descriptor_path)
    with pytest.raises(
        ValueError,
        match="details path collides with classification database owner lock",
    ):
        classifier_cli._validate_classifier_paths(
            details_path=details_path,
            database_path=database_path,
            descriptor_path=descriptor_path,
            descriptor=descriptor,
        )
    monkeypatch.setattr(
        classifier_cli,
        "SubscriptionLane",
        lambda **kwargs: FixedLane(),
    )

    class DatabaseMustNotOpen:
        def __init__(self, path: Path) -> None:
            raise AssertionError(f"DuckDB opened at {path}")

    monkeypatch.setattr(classifier_cli, "ViewerDatabase", DatabaseMustNotOpen)
    result = CliRunner().invoke(
        app,
        [
            "classify-failures",
            str(descriptor_path),
            "--provider",
            "provider",
            "--model",
            "model",
            "--database",
            str(database_path),
            "--details",
            str(details_path),
            "--repeats",
            "1",
        ],
    )

    assert result.exit_code == 2
    assert not details_path.exists()


def test_spawned_cli_invocations_serialize_before_opening_duckdb(
    tmp_path: Path,
) -> None:
    descriptor_path = _descriptor_file(tmp_path)
    database_path = tmp_path / "state.duckdb"
    details_path = tmp_path / "details.jsonl"
    executable = tmp_path / "pi"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, socket, struct, sys\n"
        "channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "channel.connect(os.environ['DR_CODE_TEST_CREDENTIAL'])\n"
        "with channel:\n"
        "    prompt = sys.stdin.read()\n"
        "    if 'Classify one parse failure.' in prompt:\n"
        "        family = 'parse'\n"
        "    elif 'Classify one test failure.' in prompt:\n"
        "        family = 'test'\n"
        "    else:\n"
        "        raise SystemExit('unrecognized prompt family')\n"
        "    payload = json.dumps(\n"
        "        {'pid': os.getpid(), 'ppid': os.getppid(), "
        "'family': family},\n"
        "        sort_keys=True, separators=(',', ':'),\n"
        "    ).encode()\n"
        "    channel.sendall(struct.pack('!I', len(payload)) + payload)\n"
        "    if channel.recv(1) != b'r':\n"
        "        raise SystemExit('provider was not released')\n"
        "label = ('wrong-algorithm' if family == 'test' "
        "else 'prose-no-code')\n"
        "print(json.dumps({'label': label, 'rationale': 'fixed'}))\n"
    )
    executable.chmod(0o755)
    bootstrap = tmp_path / "second-classifier.py"
    bootstrap.write_text(
        "import contextlib, errno, fcntl, os, struct, sys\n"
        "from dr_code.classifier import cli as classifier_cli\n"
        "real_output_lock = classifier_cli._classification_output_lock\n"
        "contention_fd = int(sys.argv.pop(1))\n"
        "@contextlib.contextmanager\n"
        "def probed_output_lock(path):\n"
        "    lock_path = classifier_cli._output_lock_path(path)\n"
        "    with lock_path.open('a+b') as probe:\n"
        "        try:\n"
        "            fcntl.flock(\n"
        "                probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB\n"
        "            )\n"
        "        except OSError as error:\n"
        "            if error.errno not in (errno.EACCES, errno.EAGAIN):\n"
        "                raise\n"
        "        else:\n"
        "            fcntl.flock(probe.fileno(), fcntl.LOCK_UN)\n"
        "            raise RuntimeError('classification output lock was free')\n"
        "    frame = b'contended'\n"
        "    with os.fdopen(contention_fd, 'wb', closefd=True) as channel:\n"
        "        channel.write(struct.pack('!I', len(frame)) + frame)\n"
        "    with real_output_lock(path):\n"
        "        yield\n"
        "classifier_cli._classification_output_lock = probed_output_lock\n"
        "from dr_code.viewer.cli import app\n"
        "app()\n"
    )
    arguments = [
        "classify-failures",
        str(descriptor_path),
        "--provider",
        "provider",
        "--model",
        "model",
        "--database",
        str(database_path),
        "--details",
        str(details_path),
        "--repeats",
        "1",
        "--parse-limit",
        "1",
        "--test-limit",
        "1",
        "--concurrency",
        "1",
    ]
    socket_directory = tempfile.TemporaryDirectory(
        prefix="dr-code-cli-",
        dir="/tmp",
    )
    socket_path = Path(socket_directory.name) / "provider.sock"
    environment = {
        **os.environ,
        "DR_CODE_TEST_CREDENTIAL": str(socket_path),
        "PATH": f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    listener: socket.socket | None = None
    contention_read_fd = -1
    contention_write_fd = -1
    first: subprocess.Popen[str] | None = None
    second: subprocess.Popen[str] | None = None
    provider_connections: list[socket.socket] = []
    provider_pids: dict[socket.socket, int] = {}
    provider_frames: list[dict[str, object]] = []
    released_connections: set[socket.socket] = set()
    blocked_provider_groups: set[int] = set()
    collected_processes: set[int] = set()
    first_stdout = ""
    first_stderr = ""
    second_stdout = ""
    second_stderr = ""
    try:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(socket_path))
        listener.listen(2)
        listener.settimeout(10)
        contention_read_fd, contention_write_fd = os.pipe()
        first = subprocess.Popen(
            [sys.executable, "-m", "dr_code.viewer", *arguments],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        first_connection, first_frame = _accept_provider_frame(listener)
        provider_connections.append(first_connection)
        provider_frames.append(first_frame)
        first_provider_pid = _provider_pid(first_frame)
        provider_pids[first_connection] = first_provider_pid
        blocked_provider_groups.add(first_provider_pid)

        second = subprocess.Popen(
            [
                sys.executable,
                str(bootstrap),
                str(contention_write_fd),
                *arguments,
            ],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            pass_fds=(contention_write_fd,),
        )
        os.close(contention_write_fd)
        contention_write_fd = -1
        assert _read_pipe_frame(contention_read_fd) == b"contended"

        _release_provider(first_connection)
        released_connections.add(first_connection)
        blocked_provider_groups.remove(first_provider_pid)

        second_connection, second_frame = _accept_provider_frame(listener)
        provider_connections.append(second_connection)
        provider_frames.append(second_frame)
        second_provider_pid = _provider_pid(second_frame)
        provider_pids[second_connection] = second_provider_pid
        blocked_provider_groups.add(second_provider_pid)
        _release_provider(second_connection)
        released_connections.add(second_connection)
        blocked_provider_groups.remove(second_provider_pid)

        first_stdout, first_stderr = first.communicate(timeout=20)
        collected_processes.add(first.pid)
        second_stdout, second_stderr = second.communicate(timeout=20)
        collected_processes.add(second.pid)
    finally:
        if contention_write_fd >= 0:
            os.close(contention_write_fd)
        if contention_read_fd >= 0:
            os.close(contention_read_fd)
        for connection in provider_connections:
            if connection not in released_connections:
                try:
                    _release_provider(connection)
                except OSError:
                    pass
                else:
                    released_connections.add(connection)
                    provider_pid = provider_pids.get(connection)
                    if provider_pid is not None:
                        blocked_provider_groups.discard(provider_pid)
            connection.close()
        if listener is not None:
            listener.close()
        socket_path.unlink(missing_ok=True)
        socket_directory.cleanup()
        for process_group_id in blocked_provider_groups:
            _signal_process_group(process_group_id, signal.SIGKILL)
        for process in (first, second):
            if process is not None and process.pid not in collected_processes:
                _terminate_and_collect_process_group(process)

    assert first is not None
    assert second is not None
    assert first.returncode == 0, first_stderr
    assert second.returncode == 0, second_stderr
    first_summary = json.loads(first_stdout)
    second_summary = json.loads(second_stdout)
    assert first_summary["classified"] == 2
    assert second_summary["classified"] == 0
    assert second_summary["resumed"] == 2
    assert (
        first_summary["experiment_identity"]
        == second_summary["experiment_identity"]
    )
    assert len(provider_frames) == 2
    assert [frame["family"] for frame in provider_frames] == ["parse", "test"]
    assert {frame["ppid"] for frame in provider_frames} == {first.pid}
    assert len({frame["pid"] for frame in provider_frames}) == 2


def _accept_provider_frame(
    listener: socket.socket,
) -> tuple[socket.socket, dict[str, object]]:
    connection, _ = listener.accept()
    try:
        connection.settimeout(10)
        payload = json.loads(_read_socket_frame(connection))
        if not isinstance(payload, dict):
            raise AssertionError("provider frame must be a JSON object")
    except BaseException:
        connection.close()
        raise
    return connection, payload


def _provider_pid(frame: dict[str, object]) -> int:
    pid = frame.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise AssertionError("provider frame must contain a positive PID")
    return pid


def _read_socket_frame(connection: socket.socket) -> bytes:
    length = struct.unpack("!I", _read_socket_exact(connection, 4))[0]
    if not 0 < length <= 4096:
        raise AssertionError("provider frame has invalid length")
    return _read_socket_exact(connection, length)


def _read_socket_exact(connection: socket.socket, length: int) -> bytes:
    payload = bytearray()
    while len(payload) < length:
        chunk = connection.recv(length - len(payload))
        if not chunk:
            raise AssertionError("provider frame ended early")
        payload.extend(chunk)
    return bytes(payload)


def _read_pipe_frame(file_descriptor: int) -> bytes:
    length = struct.unpack("!I", _read_pipe_exact(file_descriptor, 4))[0]
    if not 0 < length <= 4096:
        raise AssertionError("contention frame has invalid length")
    return _read_pipe_exact(file_descriptor, length)


def _read_pipe_exact(file_descriptor: int, length: int) -> bytes:
    payload = bytearray()
    while len(payload) < length:
        readable, _, _ = select.select([file_descriptor], [], [], 10)
        if not readable:
            raise TimeoutError("contention frame was not received")
        chunk = os.read(file_descriptor, length - len(payload))
        if not chunk:
            raise AssertionError("contention frame ended early")
        payload.extend(chunk)
    return bytes(payload)


def _release_provider(connection: socket.socket) -> None:
    connection.sendall(b"r")


def _signal_process_group(process_group_id: int, signal_number: int) -> None:
    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        pass


def _terminate_and_collect_process_group(
    process: subprocess.Popen[str],
) -> None:
    _signal_process_group(process.pid, signal.SIGTERM)
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        _signal_process_group(process.pid, signal.SIGKILL)
        process.communicate(timeout=5)
