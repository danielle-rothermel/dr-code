"""Example policy-consumer contracts.

``derive_outcome`` below is the example *consumer*: it derives a
``SubmissionOutcome``-equivalent verdict from a ``code_test`` record.
Facts stay in records; thresholds and verdicts stay in the consumer.

The contract is **outcome parity** with
``dr_code.humaneval.scoring.score_humaneval_submission``: for a submission that
extracts cleanly to code, the consumer's outcome over the ``code_test`` record
equals scoring's outcome over the evaluation.

Only the evaluation-derived outcomes a ``code_test`` record can carry are
tested against scoring (PASSED, TESTS_FAILED, TIMED_OUT,
EVALUATION_INCOMPLETE). Pre-extraction outcomes (EMPTY_SUBMISSION,
EXTRACTION_FAILED) are upstream of ``code_test`` and out of its record
scope.

``NO_TOP_LEVEL_FUNCTIONS`` is derivable from a ``code_test`` record but is
*not* asserted against scoring: preprocessing filters candidates defining no
top-level function out of the candidate set, so a submission with none never
reaches evaluation and scoring reports the earlier, more specific
extraction failure. The consumer policy still derives the outcome from a
record whose ``function_count`` is zero — a record built over a trace whose
code was not produced by that filtering — which is what the direct
derivation test below covers.
"""

from __future__ import annotations

import pytest

from dr_code.humaneval.scoring import (
    SubmissionOutcome,
    score_humaneval_submission,
)
from dr_code.core.execution.sandbox import (
    SandboxCompletedProcess,
    SandboxTimeoutError,
)
from dr_code.metrics.names import MetricName
from dr_code.metrics.records import MeasuredRecord, MetricRecord, MetricScalar


# ---------------------------------------------------------------------------
# The example consumer policy: derived from a neutral code-test record.
# ---------------------------------------------------------------------------


def derive_outcome(record: MetricRecord) -> SubmissionOutcome:
    """Derive the existing HumanEval outcome taxonomy from execution facts."""

    if record.identity.question.metric is not MetricName.CODE_TEST:
        raise ValueError("derive_outcome requires a code_test record")
    if not isinstance(record, MeasuredRecord):
        raise ValueError("derive_outcome requires a measured record")

    facts = {fact.name: fact.value for fact in record.facts}
    function_count = _integer_fact(facts, "function_count")
    if function_count == 0:
        return SubmissionOutcome.NO_TOP_LEVEL_FUNCTIONS

    failed_count = _integer_fact(facts, "failed_count")
    error_count = _integer_fact(facts, "error_count")
    timeout_count = _integer_fact(facts, "timeout_count")
    failure_count = failed_count + error_count + timeout_count
    coverage_complete = _boolean_fact(facts, "coverage_complete")

    if coverage_complete and failure_count == 0:
        return SubmissionOutcome.PASSED
    if timeout_count:
        return SubmissionOutcome.TIMED_OUT
    if not coverage_complete and failure_count == 0:
        return SubmissionOutcome.EVALUATION_INCOMPLETE
    return SubmissionOutcome.TESTS_FAILED


def _integer_fact(facts: dict[str, MetricScalar], key: str) -> int:
    value = facts.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"code_test record requires non-negative integer fact {key!r}"
        )
    return value


def _boolean_fact(facts: dict[str, MetricScalar], key: str) -> bool:
    value = facts.get(key)
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
    code_test = [
        r
        for r in records
        if r.identity.question.metric is MetricName.CODE_TEST
    ]
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


def test_no_top_level_functions_derived_from_record(
    task, local_runner, code_test_trace
) -> None:
    # Derived from the record's own facts, not asserted against scoring:
    # extraction rejects a candidate defining no top-level function, so
    # scoring never evaluates this submission.
    submission = "x = 1\n"  # compiles, no top-level functions
    record = _code_test_record(
        code_test_trace(submission, task), runner=local_runner
    )
    assert derive_outcome(record).value == (
        SubmissionOutcome.NO_TOP_LEVEL_FUNCTIONS.value
    )


def test_scoring_reports_extraction_failure_without_top_level_functions(
    task, local_runner
) -> None:
    # The counterpart claim: the outcome scoring reports for the same
    # submission is the extraction failure that preempted evaluation.
    result = score_humaneval_submission(
        raw_submission="x = 1\n", task=task, run_in_sandbox=local_runner
    )
    assert result.outcome is SubmissionOutcome.EXTRACTION_FAILED


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
    names = {fact.name for fact in record.facts}
    assert "outcome" not in names
    assert "score" not in names
    assert "pass_at_k" not in names
    assert "passed" not in names


def test_derive_outcome_rejects_negative_counts() -> None:
    """A corrupt or tampered record cannot cancel failures with negative
    counts (failed_count=-1 + error_count=1 would otherwise read as zero
    failures and derive PASSED)."""
    from dr_code.metrics import (
        MetricFact,
        MetricFactUnit,
        MetricQuestion,
        MetricQuestionCoordinate,
        MetricRecordIdentity,
        MetricsDefinition,
        MetricsDefinitionCoordinate,
    )
    from dr_code.trace import EXTERNAL_PRODUCER

    definition = MetricsDefinition(
        definition_id="policy",
        version="1",
        questions=(MetricQuestion(metric=MetricName.CODE_TEST, on="input"),),
    )
    record = MeasuredRecord(
        identity=MetricRecordIdentity(
            question=MetricQuestionCoordinate.of(definition.questions[0]),
            metric_version="1",
            producer=EXTERNAL_PRODUCER,
            metrics_definition=MetricsDefinitionCoordinate.of(definition),
        ),
        facts=(
            MetricFact(
                name="function_count", value=1, unit=MetricFactUnit.COUNT
            ),
            MetricFact(
                name="failed_count", value=-1, unit=MetricFactUnit.COUNT
            ),
            MetricFact(name="error_count", value=1, unit=MetricFactUnit.COUNT),
            MetricFact(
                name="timeout_count", value=0, unit=MetricFactUnit.COUNT
            ),
            MetricFact(
                name="coverage_complete",
                value=True,
                unit=MetricFactUnit.BOOLEAN,
            ),
        ),
    )
    with pytest.raises(ValueError, match="non-negative"):
        derive_outcome(record)
