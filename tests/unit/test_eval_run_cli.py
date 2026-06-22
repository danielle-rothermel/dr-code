"""Unit tests for eval_run lifecycle CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_eval_run_module():
    path = _REPO_ROOT / "scripts" / "eval_run.py"
    spec = importlib.util.spec_from_file_location("eval_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_start_passes_selected_stages_to_lifecycle(monkeypatch) -> None:
    cli_mod = _load_eval_run_module()
    calls = []

    def start_eval_workers(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(run_id=kwargs["run_id"], pids=[123, 456])

    monkeypatch.setattr(cli_mod, "start_eval_workers", start_eval_workers)

    result = CliRunner().invoke(
        cli_mod.app,
        [
            "start",
            "--run-id",
            "eval-1",
            "--stage",
            "parse",
            "--workers",
            "parse=1,test=0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "run_id": "eval-1",
            "workers": "parse=1,test=0",
            "stages": ["parse"],
            "handlers_module": "dr_code.pipeline.handlers",
        }
    ]
    assert "pids=123,456" in result.output


def test_wait_passes_terminal_target_without_stop(monkeypatch) -> None:
    cli_mod = _load_eval_run_module()
    calls = []

    def wait_for_eval_run(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(run_id="eval-1", status=_status())

    monkeypatch.setattr(cli_mod, "wait_for_eval_run", wait_for_eval_run)

    result = CliRunner().invoke(
        cli_mod.app,
        ["wait", "--run-id", "eval-1", "--target", "terminal"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "run_id": "eval-1",
            "target": "terminal",
            "timeout": None,
            "poll_interval": 1.0,
        }
    ]


def test_status_can_print_json(monkeypatch) -> None:
    cli_mod = _load_eval_run_module()

    monkeypatch.setattr(
        cli_mod,
        "get_eval_status",
        lambda run_id: SimpleNamespace(run_id=run_id, status=_status()),
    )

    result = CliRunner().invoke(
        cli_mod.app,
        ["status", "--run-id", "eval-1", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip() == '{"run_id":"eval-1"}'


def test_run_uses_default_worker_spec_and_lifecycle(monkeypatch) -> None:
    cli_mod = _load_eval_run_module()
    calls = []
    pipeline_result = SimpleNamespace(
        run_id="eval-1",
        expected_jobs=2,
        proof_report=SimpleNamespace(),
        export_paths=SimpleNamespace(run_dir=Path("exports/runs/eval-1")),
    )

    def run_eval_once(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(pipeline_result=pipeline_result)

    monkeypatch.setattr(cli_mod, "run_eval_once", run_eval_once)
    monkeypatch.setattr(cli_mod, "echo_run_metadata", lambda **kwargs: None)
    monkeypatch.setattr(cli_mod, "echo_proof_summary", lambda result: None)

    result = CliRunner().invoke(
        cli_mod.app,
        ["run", "--run-id", "eval-1", "--skip-preflight"],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "mode": "in-process",
            "attempts_path": None,
            "dump_dir": cli_mod.DEFAULT_DUMP_DIR,
            "task_indices": list(cli_mod.DEFAULT_PROOF_INDICES),
            "limit_per_task": None,
            "workers": "parse=8,test=8",
            "run_id": "eval-1",
            "handlers_module": "dr_code.pipeline.handlers",
            "completion_timeout": 28800.0,
            "output_root": Path("exports/runs"),
            "skip_preflight": True,
            "overwrite": False,
        }
    ]


def _status():
    return SimpleNamespace(
        terminal_jobs=1,
        expected_jobs=2,
        is_complete=False,
        stages=[
            SimpleNamespace(
                stage="parse",
                completed_jobs=1,
                expected_jobs=2,
                in_flight_jobs=0,
                input_queue=SimpleNamespace(ready_messages=1),
                workers=[],
            )
        ],
        model_dump_json=lambda: '{"run_id":"eval-1"}',
    )
