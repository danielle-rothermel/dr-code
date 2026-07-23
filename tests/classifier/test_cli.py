from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dr_code.classifier import cli as classifier_cli
from dr_code.viewer import cli
from viewer.helpers import write_bundle


class FixedLane:
    name = "mock-lane"
    model = "mock-model"

    def complete(self, prompt: str) -> str:
        return json.dumps({"label": "prose-no-code", "rationale": "fixed"})


def _write_descriptor(tmp_path: Path, descriptor) -> Path:
    payload = {
        "label": descriptor.label,
        "corpus": str(descriptor.corpus_path),
        "preprocessing": str(descriptor.preprocessing_manifest_path.parent),
        "candidate_evaluation": str(
            descriptor.evaluation_manifest_path.parent
        ),
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload))
    return path


def test_classify_failures_cli_runs_end_to_end(
    tmp_path: Path, monkeypatch
) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    run_json = _write_descriptor(tmp_path, descriptor)
    database = tmp_path / "state.duckdb"
    details = tmp_path / "details.jsonl"

    monkeypatch.setattr(
        classifier_cli.PiLane,
        "for_lane",
        classmethod(lambda cls, lane, *, model=None: FixedLane()),
    )

    result = CliRunner().invoke(
        cli.app,
        [
            "classify-failures",
            "--run",
            f"fixture={run_json}",
            "--database",
            str(database),
            "--details",
            str(details),
            "--repeats",
            "3",
            "--no-tests",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["parse_classified"] == 5
    assert payload["tasks_written"] >= 1
    assert details.is_file()
    assert len(details.read_text().splitlines()) == 5


def test_classify_failures_cli_rejects_bad_run_option(
    tmp_path: Path,
) -> None:
    result = CliRunner().invoke(
        cli.app,
        ["classify-failures", "--run", "no-equals-sign"],
    )
    assert result.exit_code != 0


def test_classify_failures_cli_rejects_unknown_lane(
    tmp_path: Path,
) -> None:
    descriptor = write_bundle(tmp_path / "bundle")
    run_json = _write_descriptor(tmp_path, descriptor)
    result = CliRunner().invoke(
        cli.app,
        [
            "classify-failures",
            "--run",
            f"fixture={run_json}",
            "--lane",
            "gpt-lane",
        ],
    )
    assert result.exit_code != 0
