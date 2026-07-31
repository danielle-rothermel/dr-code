from __future__ import annotations

import json
import os
import subprocess
import sys
import time
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
    marker_path = tmp_path / "provider-started"
    release_path = tmp_path / "provider-release"
    executable = tmp_path / "pi"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys, time\n"
        "marker = pathlib.Path(os.environ['DR_CODE_TEST_CREDENTIAL'])\n"
        "release = pathlib.Path(os.environ['DR_CODE_RELEASE_CREDENTIAL'])\n"
        "marker.touch()\n"
        "while not release.exists():\n"
        "    time.sleep(0.01)\n"
        "prompt = sys.argv[-1]\n"
        "label = ('wrong-algorithm' if "
        "'Classify one test failure.' in prompt else 'prose-no-code')\n"
        "print(json.dumps({'label': label, 'rationale': 'fixed'}))\n"
    )
    executable.chmod(0o755)
    environment = {
        **os.environ,
        "DR_CODE_TEST_CREDENTIAL": str(marker_path),
        "DR_CODE_RELEASE_CREDENTIAL": str(release_path),
        "PATH": f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    command = [
        sys.executable,
        "-m",
        "dr_code.viewer",
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
    first = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    second: subprocess.Popen[str] | None = None
    try:
        deadline = time.monotonic() + 10
        while not marker_path.exists():
            if first.poll() is not None:
                stdout, stderr = first.communicate()
                pytest.fail(
                    f"first CLI exited before provider start: {stdout} {stderr}"
                )
            if time.monotonic() >= deadline:
                pytest.fail("first CLI did not reach provider")
            time.sleep(0.01)

        second = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.4)
        assert second.poll() is None

        release_path.touch()
        first_stdout, first_stderr = first.communicate(timeout=20)
        second_stdout, second_stderr = second.communicate(timeout=20)
    finally:
        for process in (first, second):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)

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
