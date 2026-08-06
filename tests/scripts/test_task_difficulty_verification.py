from __future__ import annotations

import importlib.util
import logging
import sys
import threading
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest

from _executor_stubs import local_python_executor

_ROOT = Path(__file__).parents[2]
_SCRIPT_DIRECTORY = _ROOT / "scripts" / "verification" / "task_difficulty"
sys.path.insert(0, str(_SCRIPT_DIRECTORY))


def _load_script(filename: str) -> ModuleType:
    path = _SCRIPT_DIRECTORY / filename
    module_name = f"test_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_BUILD = _load_script("01_build_eligible_corpus.py")
_SAMPLE = _load_script("02_select_balanced_sample.py")
_EVALUATE = _load_script("03_evaluate_sample.py")
_SUMMARIZE = _load_script("04_summarize_results.py")
_SETTINGS = _load_script("workflow_settings.py")


def _evaluation_settings(
    *,
    worker_count: int = 16,
    timeout_seconds: float = 120.0,
):
    return _SETTINGS.EvaluationSettings(
        worker_count=worker_count,
        timeout_seconds=timeout_seconds,
    )


def test_workflow_logging_uses_supported_percent_placeholders() -> None:
    for path in _SCRIPT_DIRECTORY.glob("*.py"):
        assert "%," not in path.read_text()


def _generation_row(
    sample_id: str,
    *,
    encoder_model: str | None,
    encoder_output: str | None,
    encoder_user_prompt: str | None,
    decoder_output: str = "def f(x):\n    return x\n",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "source_kind": _SETTINGS.SOURCE_KIND,
        "task_id": "HumanEval/0",
        "model": "model-a",
        "encoder_model": encoder_model,
        "decoder_model": "model-a",
        "encoder_output": encoder_output,
        "encoder_user_prompt": encoder_user_prompt,
        "decoder_output": decoder_output,
    }


def test_classification_keeps_three_complete_settings() -> None:
    corpus = pl.DataFrame(
        [
            _generation_row(
                "direct",
                encoder_model=None,
                encoder_output=None,
                encoder_user_prompt=None,
            ),
            _generation_row(
                "enc-no-budget",
                encoder_model="model-a",
                encoder_output="description",
                encoder_user_prompt='{"code":"def f(): pass"}',
            ),
            _generation_row(
                "enc-budget",
                encoder_model="model-a",
                encoder_output="short description",
                encoder_user_prompt=(
                    '{"code":"def f(): pass","max_characters":50}'
                ),
            ),
            _generation_row(
                "incomplete-encoder",
                encoder_model="model-a",
                encoder_output=None,
                encoder_user_prompt=(
                    '{"code":"def f(): pass","max_characters":50}'
                ),
            ),
            _generation_row(
                "blank",
                encoder_model=None,
                encoder_output=None,
                encoder_user_prompt=None,
                decoder_output="  ",
            ),
        ]
    )

    classified = _BUILD.classify_generation_rows(corpus)

    assert classified.select(
        ["sample_id", "generation_mode", "budget_mode", "max_characters"]
    ).to_dicts() == [
        {
            "sample_id": "direct",
            "generation_mode": "direct",
            "budget_mode": "no_budget",
            "max_characters": None,
        },
        {
            "sample_id": "enc-no-budget",
            "generation_mode": "enc_dec",
            "budget_mode": "no_budget",
            "max_characters": None,
        },
        {
            "sample_id": "enc-budget",
            "generation_mode": "enc_dec",
            "budget_mode": "budget",
            "max_characters": 50,
        },
    ]


def test_preprocessing_retains_compilable_function_candidates(
    tmp_path: Path,
) -> None:
    source = "```python\ndef f(x):\n    return x + 1\n```"
    logger = logging.getLogger("test_preprocessing_candidates")
    candidates = _BUILD.preprocess_distinct_outputs(
        [source],
        cache_path=tmp_path / "cache.sqlite3",
        logger=logger,
    )
    rows = pl.DataFrame(
        [
            {
                **_generation_row(
                    "sample",
                    encoder_model=None,
                    encoder_output=None,
                    encoder_user_prompt=None,
                    decoder_output=source,
                ),
                "generation_mode": "direct",
                "budget_mode": "no_budget",
                "max_characters": None,
                "model_key": "model-a",
            }
        ]
    )

    eligible, summary = _BUILD.attach_preprocessing_results(rows, candidates)

    assert eligible.height == 1
    assert eligible.item(0, "candidate_count") >= 1
    assert any(
        candidate.startswith("def f")
        for candidate in eligible.item(0, "code_candidates")
    )
    assert summary.item(0, "eligible_rows") == 1
    assert (
        eligible.item(0, "preprocessing_definition_id")
        == "exhaustive-function-candidates"
    )


def test_sampling_selects_one_stable_row_per_cell() -> None:
    rows = []
    for task_id in ("HumanEval/0", "HumanEval/1"):
        for generation_mode, budget_mode in _SETTINGS.SETTINGS:
            for model in _SETTINGS.MODEL_ROSTER:
                for repeat in range(2):
                    rows.append(
                        {
                            "sample_id": (
                                f"{task_id}-{generation_mode}-{budget_mode}-"
                                f"{model}-{repeat}"
                            ),
                            "task_id": task_id,
                            "generation_mode": generation_mode,
                            "budget_mode": budget_mode,
                            "model_key": model,
                            "code_candidates": ["def f():\n    return 1\n"],
                            "candidate_count": 1,
                        }
                    )
    eligible = pl.DataFrame(rows)

    first, coverage = _SAMPLE.select_balanced_sample(eligible)
    second, _ = _SAMPLE.select_balanced_sample(eligible.reverse())

    assert first.height == 18
    assert coverage.get_column("selected").all()
    assert (
        first.get_column("sample_id").to_list()
        == second.get_column("sample_id").to_list()
    )


def test_selected_ground_truth_candidate_passes_full_metric() -> None:
    task = _EVALUATE._load_tasks(  # noqa: SLF001
        _SETTINGS.HUMANEVAL_SNAPSHOT,
        ["HumanEval/0"],
    )["HumanEval/0"]
    selected = pl.DataFrame(
        [
            {
                "sample_id": "ground-truth",
                "task_id": task.task_id,
                "generation_mode": "direct",
                "budget_mode": "no_budget",
                "model_key": "fixture",
                "code_candidates": [task.ground_truth_code],
                "candidate_count": 1,
            }
        ]
    )

    results = _EVALUATE.evaluate_task_rows(
        selected,
        task,
        executor=local_python_executor(),
    )

    assert results.item(0, "metric_status") == "measured"
    assert results.item(0, "candidate_passed") is True
    assert results.item(0, "metrics_definition_id") == (
        "directional-humaneval-task-difficulty"
    )


def test_evaluation_cli_overrides_workers_and_timeout() -> None:
    settings = _SETTINGS.parse_evaluation_args(
        "test parser", ["--workers", "24", "--timeout-seconds", "45.5"]
    )

    assert settings == _evaluation_settings(
        worker_count=24,
        timeout_seconds=45.5,
    )


def test_evaluation_paths_are_scoped_to_effective_settings() -> None:
    first = _SETTINGS.evaluation_paths(
        _evaluation_settings(worker_count=16, timeout_seconds=120.0)
    )
    same = _SETTINGS.evaluation_paths(
        _evaluation_settings(worker_count=16, timeout_seconds=120)
    )
    different_workers = _SETTINGS.evaluation_paths(
        _evaluation_settings(worker_count=8, timeout_seconds=120.0)
    )
    different_timeout = _SETTINGS.evaluation_paths(
        _evaluation_settings(worker_count=16, timeout_seconds=30.0)
    )

    assert first == same
    assert first.root.name == "workers-16_timeout-120"
    assert different_workers.root != first.root
    assert different_timeout.root != first.root


def test_evaluator_raises_low_open_file_soft_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = [256, 65_536]
    updates: list[tuple[int, int]] = []

    def getrlimit(_resource: int) -> tuple[int, int]:
        return (current[0], current[1])

    def setrlimit(_resource: int, limits: tuple[int, int]) -> None:
        updates.append(limits)
        current[:] = limits

    monkeypatch.setattr(_EVALUATE.resource, "getrlimit", getrlimit)
    monkeypatch.setattr(_EVALUATE.resource, "setrlimit", setrlimit)

    effective = _EVALUATE._ensure_open_file_limit(  # noqa: SLF001
        32,
        logging.getLogger("test_open_file_limit"),
    )

    assert effective == 4096
    assert updates == [(4096, 65_536)]


def test_evaluator_preserves_sufficient_open_file_soft_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _EVALUATE.resource,
        "getrlimit",
        lambda _resource: (8192, 65_536),
    )
    updates: list[tuple[int, int]] = []
    monkeypatch.setattr(
        _EVALUATE.resource,
        "setrlimit",
        lambda _resource, limits: updates.append(limits),
    )

    effective = _EVALUATE._ensure_open_file_limit(  # noqa: SLF001
        32,
        logging.getLogger("test_open_file_limit"),
    )

    assert effective == 8192
    assert updates == []


def test_evaluator_fails_before_execution_when_hard_limit_is_too_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _EVALUATE.resource,
        "getrlimit",
        lambda _resource: (256, 1024),
    )

    with pytest.raises(SystemExit, match=r"ulimit -n 4096"):
        _EVALUATE._ensure_open_file_limit(  # noqa: SLF001
            32,
            logging.getLogger("test_open_file_limit"),
        )


def test_metrics_definition_uses_requested_timeout() -> None:
    definition = _EVALUATE._metrics_definition(45.5)  # noqa: SLF001

    assert definition.questions[0].settings.timeout_seconds == 45.5


def test_task_jobs_run_concurrently_without_timing_as_evidence() -> None:
    task = _EVALUATE._load_tasks(  # noqa: SLF001
        _SETTINGS.HUMANEVAL_SNAPSHOT,
        ["HumanEval/0"],
    )["HumanEval/0"]
    jobs = [
        _EVALUATE._TaskJob(  # noqa: SLF001
            task_id=f"HumanEval/{index}",
            task_rows=pl.DataFrame(),
            task=task,
            runtime_executable=Path("/runtime/python"),
            runtime_identity="runtime-a",
            evaluation_settings=_evaluation_settings(),
            evaluation_paths=_SETTINGS.evaluation_paths(
                _evaluation_settings()
            ),
        )
        for index in range(3)
    ]
    barrier = threading.Barrier(len(jobs))

    def run_job(job):  # noqa: ANN001, ANN202
        barrier.wait(timeout=5)
        return _EVALUATE._TaskCompletion(  # noqa: SLF001
            task_id=job.task_id,
            generation_count=0,
            candidate_count=0,
            elapsed_seconds=0.0,
        )

    completions = [
        future.result()
        for _, future in _EVALUATE._completed_jobs(  # noqa: SLF001
            jobs,
            worker_count=3,
            run_job=run_job,
        )
    ]

    assert {completion.task_id for completion in completions} == {
        "HumanEval/0",
        "HumanEval/1",
        "HumanEval/2",
    }
    assert _SETTINGS.EVALUATION_WORKERS == 16


def test_evaluation_checkpoint_must_match_exact_candidates(
    tmp_path: Path,
) -> None:
    task_rows = pl.DataFrame(
        [
            {
                "sample_id": "sample",
                "candidate_count": 1,
                "code_candidates": ["def f():\n    return 1\n"],
            }
        ]
    )
    part_path = tmp_path / "part.parquet"
    pl.DataFrame(
        [
            {
                "sample_id": "sample",
                "candidate_index": 0,
                "candidate_source": "def f():\n    return 2\n",
                "runtime_identity": "runtime-a",
            }
        ]
    ).write_parquet(part_path)

    with pytest.raises(RuntimeError, match="current sample, runtime"):
        _EVALUATE._validate_existing_part(  # noqa: SLF001
            part_path,
            task_rows,
            "runtime-a",
            _evaluation_settings(),
        )


def test_evaluation_checkpoint_must_match_runtime(tmp_path: Path) -> None:
    source = "def f():\n    return 1\n"
    task_rows = pl.DataFrame(
        [
            {
                "sample_id": "sample",
                "candidate_count": 1,
                "code_candidates": [source],
            }
        ]
    )
    part_path = tmp_path / "part.parquet"
    pl.DataFrame(
        [
            {
                "sample_id": "sample",
                "candidate_index": 0,
                "candidate_source": source,
                "runtime_identity": "runtime-a",
            }
        ]
    ).write_parquet(part_path)

    with pytest.raises(RuntimeError, match="current sample, runtime"):
        _EVALUATE._validate_existing_part(  # noqa: SLF001
            part_path,
            task_rows,
            "runtime-b",
            _evaluation_settings(),
        )


def test_evaluation_checkpoint_must_match_evaluation_settings(
    tmp_path: Path,
) -> None:
    source = "def f():\n    return 1\n"
    task_rows = pl.DataFrame(
        [
            {
                "sample_id": "sample",
                "candidate_count": 1,
                "code_candidates": [source],
            }
        ]
    )
    part_path = tmp_path / "part.parquet"
    pl.DataFrame(
        [
            {
                "sample_id": "sample",
                "candidate_index": 0,
                "candidate_source": source,
                "runtime_identity": "runtime-a",
                "evaluation_worker_count": 16,
                "evaluation_timeout_seconds": 120.0,
            }
        ]
    ).write_parquet(part_path)
    _EVALUATE._validate_existing_part(  # noqa: SLF001
        part_path,
        task_rows,
        "runtime-a",
        _evaluation_settings(),
    )

    with pytest.raises(RuntimeError, match="evaluation settings"):
        _EVALUATE._validate_existing_part(  # noqa: SLF001
            part_path,
            task_rows,
            "runtime-a",
            _evaluation_settings(worker_count=8),
        )


def test_task_evaluation_rejects_operator_failures() -> None:
    results = pl.DataFrame(
        [
            {
                "metric_status": "operator_failure",
                "failure_type": "EvaluationHarnessError",
                "failure_message": "support dependency unavailable",
            }
        ]
    )

    with pytest.raises(RuntimeError, match="harness/operator failures"):
        _EVALUATE._require_measured_results(  # noqa: SLF001
            results,
            "HumanEval/0",
        )


def test_summary_uses_complete_generations_as_the_task_denominator() -> None:
    selected = pl.DataFrame(
        [
            {
                "sample_id": sample_id,
                "task_id": "HumanEval/0",
                "generation_mode": "direct",
                "budget_mode": "no_budget",
                "model_key": model,
                "candidate_count": 1,
            }
            for sample_id, model in (
                ("passed", "a"),
                ("failed", "b"),
                ("incomplete", "c"),
            )
        ]
    )
    candidate_results = pl.DataFrame(
        [
            {
                "sample_id": sample_id,
                "task_id": "HumanEval/0",
                "generation_mode": "direct",
                "budget_mode": "no_budget",
                "model_key": model,
                "metric_status": "measured",
                "candidate_passed": passed,
            }
            for sample_id, model, passed in (
                ("passed", "a", True),
                ("failed", "b", False),
            )
        ]
    )
    preprocessing = pl.DataFrame(
        [
            {
                "task_id": "HumanEval/0",
                "nonblank_rows": 10,
                "eligible_rows": 8,
            }
        ]
    )

    generations, _, tasks = _SUMMARIZE.summarize_results(
        selected,
        candidate_results,
        preprocessing,
    )

    assert generations.get_column("evaluation_complete").to_list() == [
        True,
        False,
        True,
    ]
    assert tasks.item(0, "evaluated_generations") == 2
    assert tasks.item(0, "test_success_rate") == 0.5
    assert tasks.item(0, "observed_extreme") == "mixed"
    assert tasks.item(0, "preprocessing_success_rate") == 0.8
