from __future__ import annotations

from pathlib import Path

from click import unstyle
from typer.testing import CliRunner

from dr_code.viewer import cli


def test_viewer_launches_one_loopback_worker(
    tmp_path: Path, monkeypatch
) -> None:
    descriptor = tmp_path / "run.json"
    descriptor.write_text("{}")
    database = tmp_path / "state.duckdb"
    service = object()
    captured: dict[str, object] = {}

    def fake_build(registrations, database_path):
        captured["registrations"] = registrations
        captured["database"] = database_path
        return service

    def fake_create_app(value, *, allowed_host):
        assert value is service
        captured["allowed_host"] = allowed_host
        return "application"

    def fake_run(application, **kwargs):
        captured["application"] = application
        captured.update(kwargs)

    monkeypatch.setattr(cli, "build_service", fake_build)
    monkeypatch.setattr(cli, "create_app", fake_create_app)
    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    result = CliRunner().invoke(
        cli.app,
        [
            "viewer",
            "--run",
            str(descriptor),
            "--database",
            str(database),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["registrations"] == [descriptor.resolve()]
    assert captured["database"] == database.resolve()
    assert captured["application"] == "application"
    assert captured["host"] == "127.0.0.1"
    assert captured["allowed_host"] == "127.0.0.1"
    assert captured["workers"] == 1


def test_viewer_rejects_non_loopback_host_before_starting(
    tmp_path: Path, monkeypatch
) -> None:
    descriptor = tmp_path / "run.json"
    descriptor.write_text("{}")
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    result = CliRunner().invoke(
        cli.app,
        [
            "viewer",
            "--run",
            str(descriptor),
            "--host",
            "0.0.0.0",
        ],
    )

    assert result.exit_code == 2
    assert "loopback" in result.output
    assert called is False


def test_viewer_requires_explicit_named_descriptor() -> None:
    result = CliRunner().invoke(cli.app, ["viewer"])

    assert result.exit_code == 2
    assert "Missing option '--run'" in unstyle(result.output)
