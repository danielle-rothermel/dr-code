from __future__ import annotations

from dataclasses import replace

import pytest
from dr_exec import AutoPoolCapacity, ExecutionPoolConfig

from _executor_stubs import importable_json_executor
from dr_code.evaluation import (
    BundleRecordReference,
    EvaluatedSampleRecord,
    ExecutorExecutionFailure,
    FailureClass,
)
from dr_code.evaluation._batch import _evaluate_batch_assembly
from dr_code.humaneval import (
    ANY_CANDIDATE_HUMANEVAL_SCORING_PROFILE,
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
from dr_code.humaneval.task import EvalCaseStatus
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
        pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
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
        pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
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


async def _two_candidate_record(
    *,
    first_source: str,
    second_source: str,
) -> tuple[EvaluatedSampleRecord, object]:
    """Evaluate one sample whose submission yields two ordered candidates."""

    batch_request = request()
    selected_sample = batch_request.inputs[0].data.sample.model_copy(
        update={
            "raw_input": TextArtifact(
                text=(
                    f"```python\n{first_source}\n```\n"
                    f"```python\n{second_source}\n```"
                )
            )
        }
    )
    input_item = batch_request.inputs[0]
    batch_request = batch_request.model_copy(
        update={
            "inputs": (
                input_item.model_copy(
                    update={
                        "data": input_item.data.model_copy(
                            update={"sample": selected_sample}
                        )
                    }
                ),
            )
        }
    )
    execution_cache = cache(BatchStore(), resident=2)
    placement = MemoryPlacement()
    await _evaluate_batch_assembly(
        batch_request,
        executor=importable_json_executor(),
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        placement_sink=placement,
    )
    record = placement.records[0]
    assert isinstance(record, EvaluatedSampleRecord)
    assert len(record.candidates) == 2
    return record, execution_cache


async def _one_candidate_record(
    source: str,
) -> tuple[EvaluatedSampleRecord, object]:
    """Evaluate one sample whose submission yields exactly one candidate."""

    batch_request = request()
    selected_sample = batch_request.inputs[0].data.sample.model_copy(
        update={"raw_input": TextArtifact(text=f"```python\n{source}\n```")}
    )
    input_item = batch_request.inputs[0]
    batch_request = batch_request.model_copy(
        update={
            "inputs": (
                input_item.model_copy(
                    update={
                        "data": input_item.data.model_copy(
                            update={"sample": selected_sample}
                        )
                    }
                ),
            )
        }
    )
    execution_cache = cache(BatchStore())
    placement = MemoryPlacement()
    await _evaluate_batch_assembly(
        batch_request,
        executor=importable_json_executor(),
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        placement_sink=placement,
    )
    record = placement.records[0]
    assert isinstance(record, EvaluatedSampleRecord)
    assert len(record.candidates) == 1
    return record, execution_cache


def _project(
    record: EvaluatedSampleRecord,
    profile: object,
) -> CompletedSubmissionResult | HarnessFailure:
    return project_humaneval_submission(
        record,
        HumanEvalSubmissionRequest(
            sample=record.sample.identity,
            scoring_profile=profile,  # type: ignore[arg-type]
        ),
        sample_record=_reference(),
    )


async def test_declared_reduction_decides_a_later_passing_candidate() -> None:
    record, execution_cache = await _two_candidate_record(
        first_source="def observed_load_count(_x): return 999",
        second_source="def observed_load_count(_x): return 1",
    )

    first_only = _project(record, DEFAULT_HUMANEVAL_SCORING_PROFILE)
    assert isinstance(first_only, CompletedSubmissionResult)
    assert first_only.outcome is SubmissionOutcome.TESTS_FAILED
    assert first_only.score == 0.0

    any_passes = _project(record, ANY_CANDIDATE_HUMANEVAL_SCORING_PROFILE)
    assert isinstance(any_passes, CompletedSubmissionResult)
    assert any_passes.outcome is SubmissionOutcome.PASSED
    assert any_passes.score == 1.0
    await execution_cache.close()


async def test_first_candidate_reduction_ignores_later_candidate_evidence() -> (
    None
):
    record, execution_cache = await _two_candidate_record(
        first_source="def observed_load_count(_x): return 1",
        second_source="def observed_load_count(_x): return 2",
    )
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

    projected = _project(isolated, DEFAULT_HUMANEVAL_SCORING_PROFILE)

    assert isinstance(projected, CompletedSubmissionResult)
    assert projected.outcome is SubmissionOutcome.PASSED
    await execution_cache.close()


async def test_any_candidate_reduction_scores_a_broken_measurement_as_failure() -> (
    None
):
    record, execution_cache = await _two_candidate_record(
        first_source="def observed_load_count(_x): return 999",
        second_source="def observed_load_count(_x): return 998",
    )
    broken = OperatorFailureRecord(
        identity=record.metrics[1].identity,
        failure=OperatorFailure(
            failure_type="BrokenCandidateMeasurement",
            failure_message="candidate one was never measured",
        ),
    )
    unmeasured = record.model_copy(
        update={"metrics": (record.metrics[0], broken)}
    )

    # No ordinal passes, but one is unmeasured: it might have passed, so the
    # sample is a harness failure rather than a measured zero.
    projected = _project(unmeasured, ANY_CANDIDATE_HUMANEVAL_SCORING_PROFILE)
    assert isinstance(projected, HarnessFailure)
    assert projected.failure_class is FailureClass.HARNESS
    assert projected.cause.exception_type == "BrokenCandidateMeasurement"

    # Every ordinal validly measured and none passing is a clean zero.
    measured = _project(record, ANY_CANDIDATE_HUMANEVAL_SCORING_PROFILE)
    assert isinstance(measured, CompletedSubmissionResult)
    assert measured.outcome is SubmissionOutcome.TESTS_FAILED
    assert measured.score == 0.0
    await execution_cache.close()


def _executor_failed(
    record: EvaluatedSampleRecord,
) -> EvaluatedSampleRecord:
    """Rewrite one measured record into the shape an executor failure leaves.

    An executor failure raises out of the metric operator, so the sample record
    carries an operator-failure metric beside the failed execution outcome.
    """

    (execution,) = record.executions
    return record.model_copy(
        update={
            "executions": (
                execution.model_copy(
                    update={
                        "outcome": ExecutorExecutionFailure(
                            failure_type="ExecutionPoolFailure",
                            message="the executor never ran the candidate",
                            execution_outcome=None,
                            attribution=None,
                            measurements=None,
                        )
                    }
                ),
            ),
            "metrics": (
                OperatorFailureRecord(
                    identity=record.metrics[0].identity,
                    failure=OperatorFailure(
                        failure_type="EvalHarnessError",
                        failure_message="candidate execution did not complete",
                    ),
                ),
            ),
        }
    )


async def test_executor_failure_projects_as_infrastructure() -> None:
    record, execution_cache = await _one_candidate_record(
        "def observed_load_count(_x): return 1"
    )
    failed = _executor_failed(record)

    for profile in (
        DEFAULT_HUMANEVAL_SCORING_PROFILE,
        ANY_CANDIDATE_HUMANEVAL_SCORING_PROFILE,
    ):
        projected = _project(failed, profile)
        assert isinstance(projected, HarnessFailure)
        assert projected.failure_class is FailureClass.INFRASTRUCTURE
    await execution_cache.close()


async def test_any_candidate_reduction_passes_a_solution_beside_a_helper() -> (
    None
):
    # One fenced block is one candidate, and every top-level function in it
    # becomes its own test group, so a correct solution keeps the company of
    # the helper it was written with.
    record, execution_cache = await _one_candidate_record(
        "def helper(_x):\n    return 999\n\n\n"
        "def observed_load_count(_x):\n    return 1\n"
    )
    (execution,) = record.executions
    (suite,) = execution.outcome.result.suites
    assert {group.function_name for group in suite.groups} == {
        "helper",
        "observed_load_count",
    }

    any_passes = _project(record, ANY_CANDIDATE_HUMANEVAL_SCORING_PROFILE)
    assert isinstance(any_passes, CompletedSubmissionResult)
    assert any_passes.outcome is SubmissionOutcome.PASSED
    assert any_passes.score == 1.0

    # The strict comparator profile still requires every group to pass.
    first_only = _project(record, DEFAULT_HUMANEVAL_SCORING_PROFILE)
    assert isinstance(first_only, CompletedSubmissionResult)
    assert first_only.outcome is SubmissionOutcome.TESTS_FAILED
    assert first_only.score == 0.0
    await execution_cache.close()


async def test_any_candidate_reduction_requires_one_group_to_pass_wholly() -> (
    None
):
    # No single group passes the complete suite: each function is wrong, so
    # the existential fails and the sample is a measured zero.
    record, execution_cache = await _one_candidate_record(
        "def helper(_x):\n    return 999\n\n\n"
        "def observed_load_count(_x):\n    return 998\n"
    )

    for profile in (
        DEFAULT_HUMANEVAL_SCORING_PROFILE,
        ANY_CANDIDATE_HUMANEVAL_SCORING_PROFILE,
    ):
        projected = _project(record, profile)
        assert isinstance(projected, CompletedSubmissionResult)
        assert projected.outcome is SubmissionOutcome.TESTS_FAILED
        assert projected.score == 0.0
    await execution_cache.close()


async def test_any_candidate_reduction_finds_a_passing_group_in_any_candidate() -> (
    None
):
    # The two quantifiers compose: the passing group sits in a later candidate
    # next to a failing helper.
    record, execution_cache = await _two_candidate_record(
        first_source="def observed_load_count(_x):\n    return 999",
        second_source=(
            "def helper(_x):\n    return 997\n\n\n"
            "def observed_load_count(_x):\n    return 1"
        ),
    )

    any_passes = _project(record, ANY_CANDIDATE_HUMANEVAL_SCORING_PROFILE)
    assert isinstance(any_passes, CompletedSubmissionResult)
    assert any_passes.outcome is SubmissionOutcome.PASSED
    assert any_passes.score == 1.0

    first_only = _project(record, DEFAULT_HUMANEVAL_SCORING_PROFILE)
    assert isinstance(first_only, CompletedSubmissionResult)
    assert first_only.outcome is SubmissionOutcome.TESTS_FAILED
    await execution_cache.close()


async def test_any_candidate_reduction_reports_an_unfinished_group_honestly() -> (
    None
):
    # A candidate whose solution group failed cleanly beside a group that never
    # finished is not a measured zero: the wall-time budget stopped the
    # measurement, and the unfinished group might have passed.
    record, execution_cache = await _one_candidate_record(
        "def helper(_x):\n    return 999\n\n\n"
        "def observed_load_count(_x):\n    return 998\n"
    )
    (execution,) = record.executions
    (suite,) = execution.outcome.result.suites
    solution, unfinished = (
        suite.groups
        if suite.groups[0].function_name == "helper"
        else (suite.groups[1], suite.groups[0])
    )
    stalled = unfinished.model_copy(
        update={
            "cases": tuple(
                replace(case, status=EvalCaseStatus.TIMEOUT)
                for case in unfinished.cases
            )
        }
    )
    timed_out = record.model_copy(
        update={
            "executions": (
                execution.model_copy(
                    update={
                        "outcome": execution.outcome.model_copy(
                            update={
                                "result": execution.outcome.result.model_copy(
                                    update={
                                        "suites": (
                                            suite.model_copy(
                                                update={
                                                    "groups": (
                                                        solution,
                                                        stalled,
                                                    )
                                                }
                                            ),
                                        )
                                    }
                                )
                            }
                        )
                    }
                ),
            )
        }
    )

    projected = _project(timed_out, ANY_CANDIDATE_HUMANEVAL_SCORING_PROFILE)
    assert isinstance(projected, CompletedSubmissionResult)
    assert projected.outcome is SubmissionOutcome.TIMED_OUT
    assert projected.score == 0.0
    await execution_cache.close()


@pytest.mark.parametrize(
    "profile",
    [
        DEFAULT_HUMANEVAL_SCORING_PROFILE,
        ANY_CANDIDATE_HUMANEVAL_SCORING_PROFILE,
    ],
    ids=["first_candidate", "any_candidate_passes"],
)
async def test_blank_and_unextractable_submissions_score_zero_under_every_reduction(
    profile: object,
) -> None:
    batch_request = request(2)
    inputs = tuple(
        item.model_copy(
            update={
                "data": item.data.model_copy(
                    update={
                        "sample": item.data.sample.model_copy(
                            update={"raw_input": TextArtifact(text=text)}
                        )
                    }
                )
            }
        )
        for item, text in zip(
            batch_request.inputs,
            ("   \n\n  ", "there is no code in this reply at all"),
            strict=True,
        )
    )
    batch_request = batch_request.model_copy(update={"inputs": inputs})
    execution_cache = cache(BatchStore(), resident=2)
    placement = MemoryPlacement()
    await _evaluate_batch_assembly(
        batch_request,
        executor=importable_json_executor(),
        execution_cache=execution_cache,
        pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
        placement_sink=placement,
    )

    outcomes = []
    for record in placement.records:
        projected = project_humaneval_submission(
            record,
            HumanEvalSubmissionRequest(
                sample=record.sample.identity,
                scoring_profile=profile,  # type: ignore[arg-type]
            ),
            sample_record=_reference(),
        )
        assert isinstance(projected, CompletedSubmissionResult)
        assert projected.score == 0.0
        outcomes.append(projected.outcome)

    assert set(outcomes) == {
        SubmissionOutcome.EMPTY_SUBMISSION,
        SubmissionOutcome.EXTRACTION_FAILED,
    }
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
        pool_config=ExecutionPoolConfig(capacity=AutoPoolCapacity()),
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
    assert projected.failure_class is FailureClass.HARNESS
    assert (
        projected.cause.exception_type == "UnsupportedPreprocessingDefinition"
    )

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
    assert projected.failure_class is FailureClass.HARNESS
    assert projected.cause.exception_type == "UnsupportedMetricQuestion"
    await execution_cache.close()
