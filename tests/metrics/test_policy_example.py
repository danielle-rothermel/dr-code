"""Example policy consumer contracts (plan section: ``policy_example``).

``dr_code.metrics.policy_example`` is the example *consumer*: it derives a
``SubmissionOutcome``-equivalent verdict from a ``code_test`` record (plan S5).
Facts stay in records; thresholds and verdicts stay in the consumer.

The contract is **outcome parity** with
``dr_code.humaneval.scoring.score_humaneval_submission``: for a submission that
extracts cleanly to code, the consumer's outcome over the ``code_test`` record
equals scoring's outcome over the evaluation.

Only the evaluation-derived outcomes a ``code_test`` record can carry are
tested (PASSED, TESTS_FAILED, NO_TOP_LEVEL_FUNCTIONS, TIMED_OUT,
EVALUATION_INCOMPLETE). Pre-extraction outcomes (EMPTY_SUBMISSION,
EXTRACTION_FAILED) are upstream of ``code_test`` and out of its record scope.

``dr_code.metrics`` is imported lazily inside each test.
"""

from __future__ import annotations

import pytest

from dr_code.humaneval.code_parsing import BEST_EFFORT_HUMANEVAL_PARSER_PROFILE
from dr_code.humaneval.scoring import (
    SubmissionOutcome,
    score_humaneval_submission,
)
from dr_exec import Records

from metrics.helpers import (
    code_test_trace,
    fake_executor_always,
    scripted_batch,
    wall_clock_run,
)


def _code_test_record(trace, *, executor, timeout=5.0):
    from dr_code.metrics import (
        MetricName,
        MetricQuestion,
        MetricsDefinition,
        extract_metrics,
    )

    definition = MetricsDefinition(
        definition_id="policy",
        version="1",
        questions=(
            MetricQuestion(
                metric=MetricName.CODE_TEST,
                on="input",
                settings={"timeout_seconds": timeout},
            ),
        ),
    )
    records = extract_metrics(definition, trace, executor=executor)
    code_test = [r for r in records if r.metric is MetricName.CODE_TEST]
    assert len(code_test) == 1
    return code_test[0]


def _derive_outcome(record):
    from dr_code.metrics.policy_example import derive_outcome

    return derive_outcome(record)


def _score_outcome(submission, task, *, executor, timeout=5.0) -> str:
    result = score_humaneval_submission(
        raw_submission=submission,
        task=task,
        parser_profile=BEST_EFFORT_HUMANEVAL_PARSER_PROFILE,
        timeout_seconds=timeout,
        executor=executor,
        records=Records.none(),
    )
    return result.outcome.value


def _assert_parity(submission, task, *, executor, timeout=5.0) -> None:
    """policy_example's outcome over the code_test record equals scoring's
    outcome over the same submission."""
    record = _code_test_record(
        code_test_trace(submission, task), executor=executor, timeout=timeout
    )
    consumer = _derive_outcome(record)
    oracle = _score_outcome(
        submission, task, executor=executor, timeout=timeout
    )
    assert str(consumer.value) == str(oracle)


# ---------------------------------------------------------------------------
# Outcome parity vs score_humaneval_submission.
# ---------------------------------------------------------------------------


def test_passed_outcome_parity(task, good_submission, real_executor) -> None:
    _assert_parity(good_submission, task, executor=real_executor)
    record = _code_test_record(
        code_test_trace(good_submission, task), executor=real_executor
    )
    assert _derive_outcome(record).value == SubmissionOutcome.PASSED.value


def test_tests_failed_outcome_parity(
    task, failing_submission, real_executor
) -> None:
    _assert_parity(failing_submission, task, executor=real_executor)
    record = _code_test_record(
        code_test_trace(failing_submission, task), executor=real_executor
    )
    assert _derive_outcome(record).value == (
        SubmissionOutcome.TESTS_FAILED.value
    )


def test_no_top_level_functions_outcome_parity(task, real_executor) -> None:
    submission = "x = 1\n"  # compiles, no top-level functions
    _assert_parity(submission, task, executor=real_executor)
    record = _code_test_record(
        code_test_trace(submission, task), executor=real_executor
    )
    assert _derive_outcome(record).value == (
        SubmissionOutcome.NO_TOP_LEVEL_FUNCTIONS.value
    )


def test_timed_out_outcome_parity(task, good_submission) -> None:
    executor = fake_executor_always(
        lambda call: scripted_batch(case_payloads={}, run=wall_clock_run())
    )
    _assert_parity(good_submission, task, executor=executor, timeout=1.0)
    record = _code_test_record(
        code_test_trace(good_submission, task), executor=executor, timeout=1.0
    )
    assert _derive_outcome(record).value == SubmissionOutcome.TIMED_OUT.value


def test_evaluation_incomplete_outcome_parity(task, good_submission) -> None:
    """A batch that reports only some of its cases (no failures) ⇒ incomplete."""
    from metrics.helpers import passed_payload

    def partial(call):
        first = call.request.item_ids[0]
        return scripted_batch(case_payloads={first: passed_payload()})

    executor = fake_executor_always(partial)
    _assert_parity(good_submission, task, executor=executor)
    record = _code_test_record(
        code_test_trace(good_submission, task), executor=executor
    )
    assert _derive_outcome(record).value == (
        SubmissionOutcome.EVALUATION_INCOMPLETE.value
    )


# ---------------------------------------------------------------------------
# Facts stay in records; verdicts stay in the consumer.
# ---------------------------------------------------------------------------


def test_code_test_record_carries_no_verdict_fields(
    task, good_submission, real_executor
) -> None:
    """No thresholds, verdicts, or 'best'-as-judgement in records."""
    record = _code_test_record(
        code_test_trace(good_submission, task), executor=real_executor
    )
    assert "outcome" not in record.values
    assert "score" not in record.values
    assert "pass_at_k" not in record.values
    assert "passed" not in record.values


def test_derive_outcome_rejects_negative_counts() -> None:
    """A corrupt or tampered record cannot cancel failures with negative
    counts (failed_count=-1 + error_count=1 would otherwise read as zero
    failures and derive PASSED)."""
    from dr_code.metrics import MetricName
    from dr_code.metrics.records import MetricRecord, RecordStatus

    record = MetricRecord(
        metric=MetricName.CODE_TEST,
        metric_version="1",
        on_key="input",
        producer_id="policy",
        producer_version="1",
        producer_definition_hash=None,
        metrics_definition_id="policy",
        metrics_definition_version="1",
        status=RecordStatus.MEASURED,
        values={
            "function_count": 1,
            "failed_count": -1,
            "error_count": 1,
            "timeout_count": 0,
            "coverage_complete": True,
        },
    )
    with pytest.raises(ValueError, match="non-negative"):
        _derive_outcome(record)
