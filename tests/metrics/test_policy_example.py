"""Example policy-consumer contracts.

``derive_outcome`` below is the example *consumer*: it derives a
``SubmissionOutcome``-equivalent verdict from a ``code_test`` record.
Facts stay in records; thresholds and verdicts stay in the consumer.

The contract is **outcome parity** with
``dr_code.humaneval.scoring.score_humaneval_submission``: for a submission that
extracts cleanly to code, the consumer's outcome over the ``code_test`` record
equals scoring's outcome over the evaluation.

Only the evaluation-derived outcomes a ``code_test`` record can carry are
tested (PASSED, TESTS_FAILED, NO_TOP_LEVEL_FUNCTIONS, TIMED_OUT,
EVALUATION_INCOMPLETE). Pre-extraction outcomes (EMPTY_SUBMISSION,
EXTRACTION_FAILED) are upstream of ``code_test`` and out of its record scope.

"""

from __future__ import annotations

import pytest

from dr_code.humaneval.scoring import (
    SubmissionOutcome,
    score_humaneval_submission,
)
from dr_code.humaneval.sandbox import (
    SandboxCompletedProcess,
    SandboxTimeoutError,
)
from dr_code.metrics.names import MetricName
from dr_code.metrics.records import MetricRecord, RecordStatus


# ---------------------------------------------------------------------------
# The example consumer policy: derived from a neutral code-test record.
# ---------------------------------------------------------------------------


def derive_outcome(record: MetricRecord) -> SubmissionOutcome:
    """Derive the existing HumanEval outcome taxonomy from execution facts."""

    if record.metric is not MetricName.CODE_TEST:
        raise ValueError("derive_outcome requires a code_test record")
    if record.status is not RecordStatus.MEASURED:
        raise ValueError("derive_outcome requires a measured record")

    function_count = _integer_fact(record, "function_count")
    if function_count == 0:
        return SubmissionOutcome.NO_TOP_LEVEL_FUNCTIONS

    failed_count = _integer_fact(record, "failed_count")
    error_count = _integer_fact(record, "error_count")
    timeout_count = _integer_fact(record, "timeout_count")
    failure_count = failed_count + error_count + timeout_count
    coverage_complete = _boolean_fact(record, "coverage_complete")

    if coverage_complete and failure_count == 0:
        return SubmissionOutcome.PASSED
    if timeout_count:
        return SubmissionOutcome.TIMED_OUT
    if not coverage_complete and failure_count == 0:
        return SubmissionOutcome.EVALUATION_INCOMPLETE
    return SubmissionOutcome.TESTS_FAILED


def _integer_fact(record: MetricRecord, key: str) -> int:
    value = record.values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"code_test record requires non-negative integer fact {key!r}"
        )
    return value


def _boolean_fact(record: MetricRecord, key: str) -> bool:
    value = record.values.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"code_test record requires boolean fact {key!r}")
    return value


def _code_test_record(trace, *, runner, timeout=5.0):
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
    records = extract_metrics(definition, trace, run_in_sandbox=runner)
    code_test = [r for r in records if r.metric is MetricName.CODE_TEST]
    assert len(code_test) == 1
    return code_test[0]


def _score_outcome(submission, task, *, runner, timeout=5.0) -> str:
    _ = timeout
    result = score_humaneval_submission(
        raw_submission=submission,
        task=task,
        run_in_sandbox=runner,
    )
    return result.outcome.value


def _assert_parity(
    submission, task, code_test_trace, *, runner, timeout=5.0
) -> None:
    """``derive_outcome``'s outcome over the code_test record equals scoring's
    outcome over the same submission."""
    record = _code_test_record(
        code_test_trace(submission, task), runner=runner, timeout=timeout
    )
    consumer = derive_outcome(record)
    oracle = _score_outcome(submission, task, runner=runner, timeout=timeout)
    assert str(consumer.value) == str(oracle)


# ---------------------------------------------------------------------------
# Outcome parity vs score_humaneval_submission.
# ---------------------------------------------------------------------------


def test_passed_outcome_parity(
    task, good_submission, local_runner, code_test_trace
) -> None:
    _assert_parity(good_submission, task, code_test_trace, runner=local_runner)
    record = _code_test_record(
        code_test_trace(good_submission, task), runner=local_runner
    )
    assert derive_outcome(record).value == SubmissionOutcome.PASSED.value


def test_tests_failed_outcome_parity(
    task, failing_submission, local_runner, code_test_trace
) -> None:
    _assert_parity(
        failing_submission, task, code_test_trace, runner=local_runner
    )
    record = _code_test_record(
        code_test_trace(failing_submission, task), runner=local_runner
    )
    assert derive_outcome(record).value == (
        SubmissionOutcome.TESTS_FAILED.value
    )


def test_no_top_level_functions_outcome_parity(
    task, local_runner, code_test_trace
) -> None:
    submission = "x = 1\n"  # compiles, no top-level functions
    _assert_parity(submission, task, code_test_trace, runner=local_runner)
    record = _code_test_record(
        code_test_trace(submission, task), runner=local_runner
    )
    assert derive_outcome(record).value == (
        SubmissionOutcome.NO_TOP_LEVEL_FUNCTIONS.value
    )


def test_timed_out_outcome_parity(
    task, good_submission, code_test_trace, raising_runner
) -> None:
    runner = raising_runner(SandboxTimeoutError("timed out"))
    _assert_parity(
        good_submission, task, code_test_trace, runner=runner, timeout=1.0
    )
    record = _code_test_record(
        code_test_trace(good_submission, task), runner=runner, timeout=1.0
    )
    assert derive_outcome(record).value == SubmissionOutcome.TIMED_OUT.value


def test_evaluation_incomplete_outcome_parity(
    task, good_submission, code_test_trace
) -> None:
    """Partial runner output (incomplete coverage, no failures) ⇒ incomplete."""

    def partial_runner(*, source, input_json, timeout_seconds):  # noqa: ANN001
        return SandboxCompletedProcess(
            returncode=0,
            stdout='[{"case_id": "case_0", "status": "passed"}]',
            stderr="",
        )

    _assert_parity(
        good_submission, task, code_test_trace, runner=partial_runner
    )
    record = _code_test_record(
        code_test_trace(good_submission, task), runner=partial_runner
    )
    assert derive_outcome(record).value == (
        SubmissionOutcome.EVALUATION_INCOMPLETE.value
    )


# ---------------------------------------------------------------------------
# Facts stay in records; verdicts stay in the consumer.
# ---------------------------------------------------------------------------


def test_code_test_record_carries_no_verdict_fields(
    task, good_submission, local_runner, code_test_trace
) -> None:
    """No thresholds, verdicts, or 'best'-as-judgement in records."""
    record = _code_test_record(
        code_test_trace(good_submission, task), runner=local_runner
    )
    assert "outcome" not in record.values
    assert "score" not in record.values
    assert "pass_at_k" not in record.values
    assert "passed" not in record.values


def test_derive_outcome_rejects_negative_counts() -> None:
    """A corrupt or tampered record cannot cancel failures with negative
    counts (failed_count=-1 + error_count=1 would otherwise read as zero
    failures and derive PASSED)."""
    from dr_code.metrics import MetricQuestion, MetricsDefinition
    from dr_code.trace import EXTERNAL_PRODUCER

    record = MetricRecord(
        metric=MetricName.CODE_TEST,
        metric_version="1",
        on_key="input",
        producer=EXTERNAL_PRODUCER,
        metrics_definition=MetricsDefinition(
            definition_id="policy",
            version="1",
            questions=(
                MetricQuestion(metric=MetricName.CODE_TEST, on="input"),
            ),
        ),
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
        derive_outcome(record)
