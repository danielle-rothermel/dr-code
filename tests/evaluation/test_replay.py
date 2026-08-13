from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

import pytest
from dr_exec import AutoPoolCapacity, CancelledOutcome, ExecutionPoolConfig
from dr_store import ArtifactBundlePublication, ObjectStore

from _executor_stubs import importable_json_executor, scripted_executor
from dr_code.evaluation import (
    EvalAttemptId,
    EvaluatedSampleRecord,
    PreprocessMode,
    ReplayMode,
    ReplayReady,
    ReplaySource,
    SampleData,
    SampleWithCandidatesData,
    preflight_replay,
    replay_eval_attempt,
    restore_eval_attempt,
)
from dr_code.evaluation.bundle import RestoredEvalAttempt
from dr_code.evaluation import _batch

from ._batch_builders import BatchStore, cache, request
from ._bundle_builders import publish_batch, read_limits

pytestmark = pytest.mark.asyncio


async def _source(
    root: Path,
) -> tuple[RestoredEvalAttempt, ObjectStore]:
    root.mkdir()
    result, object_store = await publish_batch(root, projections=())
    assert result.bundle_path is not None
    restored = await restore_eval_attempt(
        result.bundle_path,
        object_store=object_store,
        limits=read_limits(),
    )
    return restored, object_store


def _preflight(
    source: RestoredEvalAttempt,
    mode: ReplayMode,
    *,
    attempt_int: int = 100,
):
    context = request(
        preprocess_mode=PreprocessMode.IN_PROCESS, projections=()
    )
    return preflight_replay(
        source,
        mode,
        attempt=EvalAttemptId(attempt_id=UUID(int=attempt_int)),
        runtime=context.runtime,
        cache_namespace="tests/replay",
        run_grade=context.run_grade,
        record_placement=context.record_placement,
        projections=context.projections,
        attempt_limits=context.attempt_limits,
        window_limits=context.window_limits,
        shard_limits=context.shard_limits,
        job_budget=context.job_budget,
        preprocess_mode=PreprocessMode.IN_PROCESS,
    )


async def test_sample_replay_reconstructs_raw_input_and_auxiliary_artifacts(
    tmp_path: Path,
) -> None:
    source, _ = await _source(tmp_path / "source")

    preflight = _preflight(source, ReplayMode.SAMPLES)

    assert isinstance(preflight, ReplayReady)
    replay_input = preflight.request.inputs[0]
    assert isinstance(replay_input.data, SampleData)
    assert (
        replay_input.data.sample.raw_input
        == source.samples[0].trace.values["input"]
    )
    assert tuple(
        artifact.trace_key
        for artifact in replay_input.data.sample.auxiliary_artifacts
    ) == ("task",)

    replay_root = tmp_path / "replay"
    replay_root.mkdir()
    publication = ArtifactBundlePublication.allocate(
        replay_root, prefix="evaluation"
    )
    execution_cache = cache(BatchStore())
    try:
        result = await replay_eval_attempt(
            preflight,
            executor=importable_json_executor(),
            execution_cache=execution_cache,
            object_store=None,
            publication=publication,
            pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        )
    finally:
        await execution_cache.close()
    assert result.attempt.replay == preflight.source


async def test_materialized_candidate_replay_bypasses_preprocessing_and_persists_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, object_store = await _source(tmp_path / "source")
    preflight = _preflight(source, ReplayMode.MATERIALIZED_CANDIDATES)
    assert isinstance(preflight, ReplayReady)
    replay_input = preflight.request.inputs[0]
    assert isinstance(replay_input.data, SampleWithCandidatesData)
    assert isinstance(source.samples[0], EvaluatedSampleRecord)
    assert replay_input.data.candidates == source.samples[0].candidates

    def reject_preprocessing(*args: object, **kwargs: object) -> object:
        raise AssertionError("candidate replay must bypass preprocessing")

    monkeypatch.setattr(_batch, "bind_preprocessing", reject_preprocessing)
    replay_root = tmp_path / "replay"
    replay_root.mkdir()
    publication = ArtifactBundlePublication.allocate(
        replay_root, prefix="evaluation"
    )
    execution_cache = cache(BatchStore())
    try:
        result = await replay_eval_attempt(
            preflight,
            executor=importable_json_executor(),
            execution_cache=execution_cache,
            object_store=None,
            publication=publication,
            pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        )
    finally:
        await execution_cache.close()

    expected = ReplaySource(
        attempt=source.attempt.identity,
        mode=ReplayMode.MATERIALIZED_CANDIDATES,
    )
    assert result.attempt.replay == expected
    assert result.bundle_path is not None
    persisted = await restore_eval_attempt(
        result.bundle_path,
        object_store=object_store,
        limits=read_limits(),
    )
    assert persisted.attempt.replay == expected


async def test_malformed_sample_replay_is_rejected_before_side_effects(
    tmp_path: Path,
) -> None:
    source, _ = await _source(tmp_path / "source")
    preprocessing = source.attempt.plan.procedure.preprocessing.model_copy(
        update={"definition_id": "unsupported-definition"}
    )
    procedure = source.attempt.plan.procedure.model_copy(
        update={"preprocessing": preprocessing}
    )
    unsupported = RestoredEvalAttempt(
        attempt=source.attempt.model_copy(
            update={
                "plan": source.attempt.plan.model_copy(
                    update={"procedure": procedure}
                )
            }
        ),
        samples=source.samples,
    )
    untouched = tmp_path / "no-publication"

    with pytest.raises(ValueError, match="preprocessing"):
        _preflight(unsupported, ReplayMode.SAMPLES)
    assert not untouched.exists()


async def test_inconsistent_restored_membership_raises(tmp_path: Path) -> None:
    source, _ = await _source(tmp_path / "source")
    malformed = RestoredEvalAttempt(
        attempt=source.attempt,
        samples=(),
    )

    with pytest.raises(ValueError, match="does not cover"):
        _preflight(malformed, ReplayMode.SAMPLES)


async def test_replay_cancellation_propagates(tmp_path: Path) -> None:
    source, _ = await _source(tmp_path / "source")
    preflight = _preflight(source, ReplayMode.SAMPLES)
    assert isinstance(preflight, ReplayReady)
    cancellation_root = tmp_path / "cancelled"
    cancellation_root.mkdir()
    publication = ArtifactBundlePublication.allocate(
        cancellation_root, prefix="evaluation"
    )
    execution_cache = cache(BatchStore())
    try:
        with pytest.raises(asyncio.CancelledError):
            await replay_eval_attempt(
                preflight,
                executor=scripted_executor(outcome=CancelledOutcome()),
                execution_cache=execution_cache,
                object_store=None,
                publication=publication,
                pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
            )
    finally:
        await execution_cache.close()
