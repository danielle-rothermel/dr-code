from __future__ import annotations

import pytest
from dr_exec import ExecutionPoolConfig

from _executor_stubs import importable_json_executor
from dr_code.evaluation import BundleRecordReference, EvaluatedSampleRecord
from dr_code.evaluation._batch import _evaluate_batch_assembly
from dr_code.humaneval import (
    DEFAULT_HUMANEVAL_SCORING_PROFILE,
    CompletedSubmissionResult,
    HarnessFailure,
    HumanEvalSubmissionRequest,
    SubmissionOutcome,
    project_humaneval_submission,
    project_humaneval_submissions_batch,
    score_humaneval_submission,
    score_humaneval_submissions_batch,
)
from dr_code.metrics import OperatorFailure, OperatorFailureRecord
from dr_code.trace import TextArtifact

from evaluation._batch_builders import (
    BatchStore,
    MemoryPlacement,
    cache,
    request,
)

pytestmark = pytest.mark.asyncio


def _reference(index: int = 0) -> BundleRecordReference:
    return BundleRecordReference(
        artifact_name="sample-records-00000000.jsonl",
        record_index=index,
        record_sha256=f"{index + 1:064x}",
        schema="dr-code/sample-evaluation-record-v1",
        schema_version=1,
    )


async def test_projection_scores_authoritative_records_without_execution() -> (
    None
):
    batch_request = request()
    execution_cache = cache(BatchStore())
    placement = MemoryPlacement()
    await _evaluate_batch_assembly(
        batch_request,
        executor=importable_json_executor(),
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(),
        placement_sink=placement,
    )
    record = placement.records[0]
    assert isinstance(record, EvaluatedSampleRecord)
    projection_request = HumanEvalSubmissionRequest(
        sample=record.sample.identity,
        scoring_profile=DEFAULT_HUMANEVAL_SCORING_PROFILE,
    )

    projected = project_humaneval_submission(
        record,
        projection_request,
        sample_record=_reference(),
    )

    assert isinstance(projected, CompletedSubmissionResult)
    assert projected.outcome is SubmissionOutcome.PASSED
    assert projected.score == 1.0
    assert projected.sample_record == _reference()
    assert (
        score_humaneval_submission(
            record,
            projection_request,
            sample_record=_reference(),
        )
        == projected
    )
    with pytest.raises(TypeError):
        project_humaneval_submission(  # type: ignore[call-arg]
            record,
            projection_request,
            sample_record=_reference(),
            executor=importable_json_executor(),
        )
    await execution_cache.close()


async def test_batch_projection_preserves_request_order() -> None:
    batch_request = request(2)
    execution_cache = cache(BatchStore())
    placement = MemoryPlacement()
    await _evaluate_batch_assembly(
        batch_request,
        executor=importable_json_executor(),
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(),
        placement_sink=placement,
    )
    records = tuple(
        (record, _reference(index))
        for index, record in enumerate(placement.records)
    )
    requests = tuple(
        HumanEvalSubmissionRequest(
            sample=record.sample.identity,
            scoring_profile=DEFAULT_HUMANEVAL_SCORING_PROFILE,
        )
        for record in reversed(placement.records)
    )

    projected = project_humaneval_submissions_batch(records, requests)

    assert [item.sample for item in projected] == [
        item.sample for item in requests
    ]
    assert score_humaneval_submissions_batch(records, requests) == projected
    await execution_cache.close()


async def test_projection_classifies_only_the_first_candidate_metric_slice() -> (
    None
):
    batch_request = request()
    selected_sample = batch_request.inputs[0].sample.model_copy(
        update={
            "raw_input": TextArtifact(
                text=(
                    "```python\ndef observed_load_count(_x): return 1\n```\n"
                    "```python\ndef observed_load_count(_x): return 2\n```"
                )
            )
        }
    )
    batch_request = batch_request.model_copy(
        update={
            "inputs": (
                batch_request.inputs[0].model_copy(
                    update={"sample": selected_sample}
                ),
            )
        }
    )
    execution_cache = cache(BatchStore(), resident=1)
    placement = MemoryPlacement()
    await _evaluate_batch_assembly(
        batch_request,
        executor=importable_json_executor(),
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(),
        placement_sink=placement,
    )
    record = placement.records[0]
    assert isinstance(record, EvaluatedSampleRecord)
    assert len(record.candidates) == 2
    later_failure = OperatorFailureRecord(
        identity=record.metrics[1].identity,
        failure=OperatorFailure(
            failure_type="LaterCandidateFailure",
            failure_message="must not poison candidate zero",
        ),
    )
    isolated = record.model_copy(
        update={"metrics": (record.metrics[0], later_failure)}
    )
    projection_request = HumanEvalSubmissionRequest(
        sample=record.sample.identity,
        scoring_profile=DEFAULT_HUMANEVAL_SCORING_PROFILE,
    )

    projected = project_humaneval_submission(
        isolated,
        projection_request,
        sample_record=_reference(),
    )

    assert isinstance(projected, CompletedSubmissionResult)
    assert projected.outcome is SubmissionOutcome.PASSED
    await execution_cache.close()


async def test_projection_requires_full_profile_coordinate_and_question() -> (
    None
):
    execution_cache = cache(BatchStore())
    placement = MemoryPlacement()
    await _evaluate_batch_assembly(
        request(),
        executor=importable_json_executor(),
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(),
        placement_sink=placement,
    )
    record = placement.records[0]
    assert isinstance(record, EvaluatedSampleRecord)

    preprocessing = DEFAULT_HUMANEVAL_SCORING_PROFILE.preprocessing_definition
    wrong_preprocessing = preprocessing.model_copy(update={"steps": ()})
    wrong_profile = DEFAULT_HUMANEVAL_SCORING_PROFILE.model_copy(
        update={"preprocessing_definition": wrong_preprocessing}
    )
    projected = project_humaneval_submission(
        record,
        HumanEvalSubmissionRequest(
            sample=record.sample.identity,
            scoring_profile=wrong_profile,
        ),
        sample_record=_reference(),
    )
    assert isinstance(projected, HarnessFailure)
    assert projected.failure_class == "UnsupportedPreprocessingDefinition"

    wrong_question = DEFAULT_HUMANEVAL_SCORING_PROFILE.question.model_copy(
        update={"on_key": "other-output"}
    )
    wrong_profile = DEFAULT_HUMANEVAL_SCORING_PROFILE.model_copy(
        update={"question": wrong_question}
    )
    projected = project_humaneval_submission(
        record,
        HumanEvalSubmissionRequest(
            sample=record.sample.identity,
            scoring_profile=wrong_profile,
        ),
        sample_record=_reference(),
    )
    assert isinstance(projected, HarnessFailure)
    assert projected.failure_class == "UnsupportedMetricQuestion"
    await execution_cache.close()
