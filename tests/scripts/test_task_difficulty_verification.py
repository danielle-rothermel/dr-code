from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from uuid import UUID

import polars as pl
import pytest
from dr_exec import ExecutionPoolConfig, FixedPoolCapacity
from dr_store import (
    ArtifactBundlePublication,
    CacheEntry,
    CacheHit,
    MemoryBackend,
    ObjectReference,
    ObjectStore,
)

from _executor_stubs import importable_json_executor
from dr_code.caching import WindowedExecutionCache
from dr_code.evaluation import AttemptCompleteness, evaluate_batch

_ROOT = Path(__file__).parents[2]
_SCRIPT_DIRECTORY = _ROOT / "scripts" / "verification" / "task_difficulty"
_FIXTURE_BUNDLE = (
    _ROOT / "tests" / "fixtures" / "generation_corpus" / "human_eval"
)
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
_SUMMARIZE = _load_script("04_summarize_results.py")
_SETTINGS = _load_script("workflow_settings.py")
_LOADER = _load_script("corpus_loader.py")
_BATCH = _load_script("evaluation_batch.py")


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
    generation_mode: str,
    budget_mode: str,
    encoder_model: str | None,
    encoder_output: str | None,
    encoder_user_prompt: str | None,
    max_characters: int | None = None,
    decoder_output: str = "def f(x):\n    return x\n",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "task_id": "HumanEval/0",
        "model": "model-a",
        "encoder_model": encoder_model,
        "encoder_output": encoder_output,
        "encoder_user_prompt": encoder_user_prompt,
        "decoder_output": decoder_output,
        "generation_mode": generation_mode,
        "budget_mode": budget_mode,
        "max_characters": max_characters,
    }


def test_load_workflow_frame_reads_fixture_bundle() -> None:
    frame = _LOADER.load_workflow_frame(_FIXTURE_BUNDLE)
    assert frame.height == 6
    assert "sample_id" in frame.columns
    assert frame.filter(pl.col("sample_id") == "direct").height == 1


def test_load_workflow_frame_validates_manifest_sha256() -> None:
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        _LOADER.load_workflow_frame(
            _FIXTURE_BUNDLE,
            expected_manifest_sha256="0" * 64,
        )


def test_classification_keeps_three_complete_settings() -> None:
    corpus = _LOADER.load_workflow_frame(_FIXTURE_BUNDLE)

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
    candidates = asyncio.run(
        _BUILD.preprocess_distinct_outputs(
            [source],
            cache_path=tmp_path / "cache.sqlite3",
            logger=logger,
        )
    )
    rows = pl.DataFrame(
        [
            _generation_row(
                "sample",
                generation_mode="direct",
                budget_mode="no_budget",
                encoder_model=None,
                encoder_output=None,
                encoder_user_prompt=None,
                decoder_output=source,
            )
            | {"model_key": "model-a"}
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


def test_preprocessing_continues_after_distinct_output_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good_source = "```python\ndef f(x):\n    return x + 1\n```"
    bad_source = "def broken():\n    return 1\n"
    logger = logging.getLogger("test_preprocessing_failure_tolerance")
    original = _BUILD.preprocess_batch

    async def _preprocess_batch(texts, **kwargs):  # noqa: ANN001
        return await original(
            [text for text in texts if text != bad_source],
            **kwargs,
        )

    monkeypatch.setattr(_BUILD, "preprocess_batch", _preprocess_batch)

    results = asyncio.run(
        _BUILD.preprocess_distinct_outputs(
            [bad_source, good_source],
            cache_path=tmp_path / "cache.sqlite3",
            logger=logger,
        )
    )

    assert results[bad_source] == ()
    assert len(results[good_source]) >= 1


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


def test_batch_request_preserves_slot_order_and_limits() -> None:
    selected = pl.DataFrame(
        [
            {
                "sample_id": "sample-a",
                "task_id": "HumanEval/0",
                "generation_mode": "direct",
                "budget_mode": "no_budget",
                "model_key": "model-a",
                "decoder_output": "def f():\n    return 1\n",
                "code_candidates": ["def f():\n    return 1\n"],
                "candidate_count": 1,
            },
            {
                "sample_id": "sample-b",
                "task_id": "HumanEval/0",
                "generation_mode": "enc_dec",
                "budget_mode": "no_budget",
                "model_key": "model-b",
                "decoder_output": "def g():\n    return 2\n",
                "code_candidates": ["def g():\n    return 2\n"],
                "candidate_count": 1,
            },
        ]
    )
    runtime = _BATCH.runtime_identity_from_executor(importable_json_executor())
    attempt = _BATCH.attempt_identity("test-fingerprint")
    request = _BATCH.build_task_difficulty_batch_request(
        selected,
        snapshot_path=_SETTINGS.HUMANEVAL_SNAPSHOT,
        manifest_sha256="a" * 64,
        settings=_evaluation_settings(worker_count=2),
        runtime=runtime,
        attempt=attempt,
    )

    assert request.plan.repeat_plan.repeats == 2
    assert len(request.inputs) == 2
    assert request.inputs[0].sample.metadata.identity.sample_id == "sample-a"
    assert request.inputs[1].sample.metadata.identity.sample_id == "sample-b"
    assert request.attempt_limits.max_slots == 2
    assert request.attempt_limits.max_materialized_candidates == 2


def test_settings_fingerprint_changes_with_workers(
    tmp_path: Path,
) -> None:
    selected_path = tmp_path / "selected.parquet"
    pl.DataFrame([{"sample_id": "x"}]).write_parquet(selected_path)
    manifest = "b" * 64
    first = _BATCH.settings_fingerprint(
        settings=_evaluation_settings(worker_count=16),
        manifest_sha256=manifest,
        selected_sample_path=selected_path,
    )
    second = _BATCH.settings_fingerprint(
        settings=_evaluation_settings(worker_count=8),
        manifest_sha256=manifest,
        selected_sample_path=selected_path,
    )
    assert first != second


def test_attempt_identity_is_deterministic() -> None:
    first = _BATCH.attempt_identity("same")
    second = _BATCH.attempt_identity("same")
    assert first.attempt_id == second.attempt_id
    assert isinstance(first.attempt_id, UUID)


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
    assert first.bundle_root == first.root / "evaluation_bundles"
    assert different_workers.root != first.root
    assert different_timeout.root != first.root


def test_candidate_job_budget_scales_with_timeout() -> None:
    budget = _BATCH.candidate_job_budget(45.5)

    assert budget.wall_time_ns == 45_500_000_000
    assert budget.input_bytes == 2_097_152
    assert budget.payload_output_bytes == 1_073_741_824


class _BatchStore:
    def __init__(self) -> None:
        self.records: dict[str, tuple[str, object]] = {}

    async def get_many(self, keys, *, schema: str):  # noqa: ANN001
        return {
            key: (
                CacheHit(record=stored[1])
                if (stored := self.records.get(key)) is not None
                and stored[0] == schema
                else None
            )
            for key in keys
        }

    async def put_many(self, entries):  # noqa: ANN001
        for key, entry in entries.items():
            if isinstance(entry, CacheEntry):
                self.records[key] = (entry.schema, entry.record)
        return {
            key: ObjectReference.for_record(entry.schema, entry.record)
            for key, entry in entries.items()
        }


@pytest.mark.asyncio
async def test_evaluate_batch_exports_candidate_results_for_fixture_sample(
    tmp_path: Path,
) -> None:
    task = _BATCH.load_humaneval_tasks(
        _SETTINGS.HUMANEVAL_SNAPSHOT,
        ("HumanEval/0",),
    )["HumanEval/0"]
    selected = pl.DataFrame(
        [
            {
                "sample_id": "fixture-sample",
                "task_id": "HumanEval/0",
                "generation_mode": "direct",
                "budget_mode": "no_budget",
                "model_key": "fixture",
                "decoder_output": "",
                "code_candidates": [task.ground_truth_code],
                "candidate_count": 1,
            }
        ]
    )
    settings = _evaluation_settings(worker_count=1, timeout_seconds=30.0)
    runtime = _BATCH.runtime_identity_from_executor(importable_json_executor())
    runtime = _BATCH.runtime_identity_with_packages(
        runtime,
        {"python_version": "test"},
    )
    attempt = _BATCH.attempt_identity("fixture-eval")
    request = _BATCH.build_task_difficulty_batch_request(
        selected,
        snapshot_path=_SETTINGS.HUMANEVAL_SNAPSHOT,
        manifest_sha256="c" * 64,
        settings=settings,
        runtime=runtime,
        attempt=attempt,
    )
    publication = ArtifactBundlePublication.allocate(
        tmp_path,
        prefix="evaluation",
    )
    object_store = ObjectStore(MemoryBackend())
    execution_cache = WindowedExecutionCache(
        _BatchStore(),
        runtime=runtime,
        max_resident_entries=4,
        max_pending_checkpoint_entries=4,
    )
    try:
        result = await evaluate_batch(
            request,
            executor=importable_json_executor(),
            execution_cache=execution_cache,
            object_store=object_store,
            publication=publication,
            pool_config=ExecutionPoolConfig(
                capacity=FixedPoolCapacity(max_active_jobs=1)
            ),
        )
    finally:
        await execution_cache.close()

    assert result.attempt.completeness is AttemptCompleteness.COMPLETE
    assert result.bundle_path is not None
    runtime_json = _BATCH.runtime_identity_json(runtime)
    output_path = tmp_path / "candidate_results.parquet"
    exported = _BATCH.export_candidate_results(
        result.bundle_path,
        selected,
        output_path,
        settings=settings,
        runtime_identity_json=runtime_json,
    )
    assert exported.height >= 1
    assert "candidate_passed" in exported.columns
    assert exported.item(0, "metrics_definition_id") == (
        "directional-humaneval-task-difficulty"
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
