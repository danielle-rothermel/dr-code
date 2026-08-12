from __future__ import annotations

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
    # Register under the real stem: spawned worker processes re-import
    # pickled module-level functions by this name from _SCRIPT_DIRECTORY.
    path = _SCRIPT_DIRECTORY / filename
    module_name = path.stem
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


def test_classification_keeps_every_complete_nonblank_row() -> None:
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
        {
            "sample_id": "unresolved",
            "generation_mode": "unresolved_encoder",
            "budget_mode": "unresolved",
            "max_characters": None,
        },
    ]


def test_classification_keeps_a_novel_mode_and_budget_combination() -> None:
    """Nothing may silently exclude a populated cell from the probe."""

    corpus = pl.DataFrame(
        [
            _generation_row(
                "novel",
                generation_mode="future_mode",
                budget_mode="future_budget",
                encoder_model=None,
                encoder_output=None,
                encoder_user_prompt=None,
            )
            | {"decoder_model": "model-a"}
        ]
    )

    classified = _BUILD.classify_generation_rows(corpus)

    assert classified.get_column("sample_id").to_list() == ["novel"]
    assert classified.item(0, "generation_mode") == "future_mode"
    assert classified.item(0, "budget_mode") == "future_budget"


def test_preprocessing_retains_compilable_function_candidates() -> None:
    source = "```python\ndef f(x):\n    return x + 1\n```"
    logger = logging.getLogger("test_preprocessing_candidates")
    candidates = _BUILD.preprocess_distinct_outputs(
        [source],
        logger=logger,
        worker_count=2,
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


def test_preprocessing_continues_after_distinct_output_failure() -> None:
    good_source = "```python\ndef f(x):\n    return x + 1\n```"
    bad_source = "```python\ndef broken(:\n    return 1\n```"
    logger = logging.getLogger("test_preprocessing_failure_tolerance")

    results = _BUILD.preprocess_distinct_outputs(
        [bad_source, good_source],
        logger=logger,
        worker_count=2,
    )

    assert results[bad_source] == ()
    assert len(results[good_source]) >= 1


def test_preprocess_timeout_flag_defaults_to_ten_minutes() -> None:
    """Stage 1 carries the workflow's watchdog default, not the library's."""

    arguments = _BUILD._parse_args([])

    assert arguments.preprocess_timeout_seconds == 600.0


@pytest.mark.parametrize("opt_out", ["0", "none", "None", " none "])
def test_preprocess_timeout_flag_accepts_an_explicit_opt_out(
    opt_out: str,
) -> None:
    arguments = _BUILD._parse_args(["--preprocess-timeout-seconds", opt_out])

    assert arguments.preprocess_timeout_seconds is None


def test_preprocess_timeout_flag_accepts_a_positive_budget() -> None:
    arguments = _BUILD._parse_args(["--preprocess-timeout-seconds", "12.5"])

    assert arguments.preprocess_timeout_seconds == 12.5


@pytest.mark.parametrize("rejected", ["-1", "nan", "inf", "abc"])
def test_preprocess_timeout_flag_rejects_unusable_values(
    rejected: str,
) -> None:
    with pytest.raises(SystemExit):
        _BUILD._parse_args(["--preprocess-timeout-seconds", rejected])


def test_preprocess_timeout_env_override_reaches_the_flag_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        _SETTINGS.PREPROCESS_TIMEOUT_SECONDS_ENV,
        "45",
    )

    assert _SETTINGS.preprocess_timeout_seconds() == 45.0


def test_preprocess_timeout_env_override_can_disable_the_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_SETTINGS.PREPROCESS_TIMEOUT_SECONDS_ENV, "none")

    assert _SETTINGS.preprocess_timeout_seconds() is None


_SAMPLING_GROUPS = (
    ("direct", "no_budget", "model-a"),
    ("enc_dec", "budget", "model-b"),
    ("unresolved_encoder", "unresolved", "model-c"),
)


def _eligible_grid(task_count: int, *, repeats: int = 2) -> pl.DataFrame:
    rows = []
    for task_index in range(task_count):
        task_id = f"HumanEval/{task_index}"
        for generation_mode, budget_mode, model in _SAMPLING_GROUPS:
            for repeat in range(repeats):
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
    return pl.DataFrame(rows)


def test_sampling_membership_is_stable_for_a_seed_and_corpus() -> None:
    eligible = _eligible_grid(10)

    first, _ = _SAMPLE.select_balanced_sample(eligible, tasks_per_group=4)
    second, _ = _SAMPLE.select_balanced_sample(
        eligible.reverse(), tasks_per_group=4
    )

    assert (
        first.get_column("sample_id").to_list()
        == second.get_column("sample_id").to_list()
    )
    assert first.get_column("sampling_seed").unique().to_list() == [
        _SETTINGS.SAMPLING_SEED
    ]


def test_sampling_represents_every_populated_group() -> None:
    eligible = _eligible_grid(10)

    selected, coverage = _SAMPLE.select_balanced_sample(
        eligible, tasks_per_group=4
    )

    observed = set(
        selected.select(
            ["generation_mode", "budget_mode", "model_key"]
        ).iter_rows()
    )
    assert observed == set(_SAMPLING_GROUPS)
    assert coverage.height == len(_SAMPLING_GROUPS)
    assert coverage.get_column("selected_tasks").to_list() == [4, 4, 4]
    assert coverage.get_column("eligible_tasks").to_list() == [10, 10, 10]


def test_sampling_honors_the_tasks_per_group_budget() -> None:
    eligible = _eligible_grid(10)

    selected, _ = _SAMPLE.select_balanced_sample(eligible, tasks_per_group=3)

    assert selected.height == 3 * len(_SAMPLING_GROUPS)
    per_group = selected.group_by(
        ["generation_mode", "budget_mode", "model_key"]
    ).len()
    assert per_group.get_column("len").unique().to_list() == [3]
    assert selected.get_column("tasks_per_group").unique().to_list() == [3]


def test_sampling_budget_above_the_corpus_keeps_every_task() -> None:
    eligible = _eligible_grid(4)

    selected, _ = _SAMPLE.select_balanced_sample(eligible, tasks_per_group=40)

    assert selected.height == 4 * len(_SAMPLING_GROUPS)


def test_sampling_with_all_keeps_the_full_group_grid() -> None:
    eligible = _eligible_grid(10)

    selected, coverage = _SAMPLE.select_balanced_sample(
        eligible, tasks_per_group=None
    )

    assert selected.height == 10 * len(_SAMPLING_GROUPS)
    assert coverage.get_column("selected_tasks").to_list() == [10, 10, 10]
    assert selected.get_column("tasks_per_group").unique().to_list() == [None]


def test_sampling_rejects_a_corpus_with_no_populated_group() -> None:
    eligible = _eligible_grid(1).head(0)

    with pytest.raises(ValueError, match="no populated"):
        _SAMPLE.select_balanced_sample(eligible, tasks_per_group=4)


def test_tasks_per_group_flag_defaults_to_the_workflow_budget() -> None:
    arguments = _SAMPLE._parse_args([])

    assert arguments.tasks_per_group == _SETTINGS.SAMPLE_TASKS_PER_GROUP


@pytest.mark.parametrize("full_grid", ["0", "all", "All", " all "])
def test_tasks_per_group_flag_accepts_the_full_grid(full_grid: str) -> None:
    arguments = _SAMPLE._parse_args(["--tasks-per-group", full_grid])

    assert arguments.tasks_per_group is None


@pytest.mark.parametrize("rejected", ["-1", "abc", "2.5"])
def test_tasks_per_group_flag_rejects_unusable_values(rejected: str) -> None:
    with pytest.raises(SystemExit):
        _SAMPLE._parse_args(["--tasks-per-group", rejected])


def test_tasks_per_group_env_override_reaches_the_flag_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_SETTINGS.SAMPLE_TASKS_PER_GROUP_ENV, "7")

    assert _SETTINGS.sample_tasks_per_group() == 7


def test_tasks_per_group_env_override_can_request_every_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_SETTINGS.SAMPLE_TASKS_PER_GROUP_ENV, "all")

    assert _SETTINGS.sample_tasks_per_group() is None


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


def test_batch_request_admits_a_ragged_per_task_sample() -> None:
    """Per-group task subsetting gives tasks unequal generation counts."""

    selected = pl.DataFrame(
        [
            {
                "sample_id": sample_id,
                "task_id": task_id,
                "generation_mode": "direct",
                "budget_mode": "no_budget",
                "model_key": model_key,
                "decoder_output": "def f():\n    return 1\n",
                "code_candidates": ["def f():\n    return 1\n"],
                "candidate_count": 1,
            }
            for sample_id, task_id, model_key in (
                ("a", "HumanEval/0", "model-a"),
                ("b", "HumanEval/0", "model-b"),
                ("c", "HumanEval/1", "model-a"),
            )
        ]
    )
    runtime = _BATCH.runtime_identity_from_executor(importable_json_executor())
    attempt = _BATCH.attempt_identity("ragged-fingerprint")

    request = _BATCH.build_task_difficulty_batch_request(
        selected,
        snapshot_path=_SETTINGS.HUMANEVAL_SNAPSHOT,
        manifest_sha256="a" * 64,
        settings=_evaluation_settings(worker_count=2),
        runtime=runtime,
        attempt=attempt,
    )

    # The widest task sets the declared slot width; narrower tasks fill fewer.
    assert request.plan.repeat_plan.repeats == 2
    assert len(request.inputs) == 3
    slots = [
        (
            evaluation_input.slot.task_id,
            evaluation_input.slot.repeat_index,
        )
        for evaluation_input in request.inputs
    ]
    assert slots == [
        ("HumanEval/0", 0),
        ("HumanEval/0", 1),
        ("HumanEval/1", 0),
    ]


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


def test_read_limits_admit_one_projection_row_per_candidate() -> None:
    limits = _BATCH.evaluation_read_limits(
        sample_count=326,
        candidate_count=460,
    )

    assert limits.max_sample_records >= 460
    assert limits.max_object_reads >= 326
    # One reference shard per sample plus fixed manifest-adjacent artifacts.
    assert limits.bundle.max_artifacts >= 326 + 1


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
