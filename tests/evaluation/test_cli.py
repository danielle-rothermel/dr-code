from __future__ import annotations

import tomllib
from pathlib import Path

from typer.testing import CliRunner

from dr_code.evaluation.cli import (
    validate_preprocessing_app,
    validate_testing_app,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DECLARED_SCRIPTS = {
    "dr-code-humaneval-schema": "dr_code.humaneval.schema_cli:app",
    "dr-code-synthetic": "dr_code.synthetic.cli:app",
    "dr-code-validate-preprocessing": "dr_code.evaluation.cli:validate_preprocessing_app",
    "dr-code-validate-testing": "dr_code.evaluation.cli:validate_testing_app",
}


def test_declared_scripts_are_exactly_the_packaged_entry_points() -> None:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as file:
        pyproject = tomllib.load(file)

    assert pyproject["project"]["scripts"] == _DECLARED_SCRIPTS


def test_validation_verbs_declare_their_flow_arguments() -> None:
    runner = CliRunner()

    for app in (validate_preprocessing_app, validate_testing_app):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0, result.output
        for option in ("--request", "--run-root", "--runtime", "--workers"):
            assert option in result.output
