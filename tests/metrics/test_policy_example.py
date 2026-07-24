"""Example policy-consumer contracts.

``dr_code.metrics.policy_example`` is the example *consumer*: it derives a
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
from dr_code.execution.subprocess import (
    SubprocessCompletedProcess,
    SubprocessTimeoutError,
)

from metrics.helpers import (
    code_test_trace,
    evaluation_procedure,
    procedure_trace,
    raising_runner,
)

_PROCEDURE_HASH = "d" * 64


def _code_test_record(trace, *, runner, timeout=5.0):
    from dr_code.eval import (
        MetricQuestionBinding,
        MetricExtractionDefinition,
    )
    from dr_code.metrics import (
        MetricName,
        extract_metrics,
    )

    definition = MetricExtractionDefinition(
        definition_id="policy",
        version="1",
        questions=(
            MetricQuestionBinding(
                metric=MetricName.CODE_TEST,
                on="input",
                settings={"timeout_seconds": timeout},
            ),
        ),
    )
    metric_extraction = definition.materialize()
    procedure = evaluation_procedure(definition, metric_extraction)
    records = extract_metrics(
        procedure_trace(trace, procedure),
        metric_extraction=metric_extraction,
        evaluation_procedure=procedure,
        run_in_subprocess=runner,
    )
    code_test = [r for r in records if r.question == MetricName.CODE_TEST]
    assert len(code_test) == 1
    return code_test[0]


def _derive_outcome(record):
    from dr_code.metrics.policy_example import derive_outcome

    return derive_outcome(record)


def _score_outcome(submission, task, *, runner, timeout=5.0) -> str:
    result = score_humaneval_submission(
        raw_submission=submission,
        task=task,
        timeout_seconds=timeout,
        run_in_subprocess=runner,
    )
    return result.outcome.value


def _assert_parity(submission, task, *, runner, timeout=5.0) -> None:
    """policy_example's outcome over the code_test record equals scoring's
    outcome over the same submission."""
    record = _code_test_record(
        code_test_trace(submission, task), runner=runner, timeout=timeout
    )
    consumer = _derive_outcome(record)
    oracle = _score_outcome(submission, task, runner=runner, timeout=timeout)
    assert str(consumer.value) == str(oracle)


# ---------------------------------------------------------------------------
# Outcome parity vs score_humaneval_submission.
# ---------------------------------------------------------------------------


def test_passed_outcome_parity(task, good_submission, local_runner) -> None:
    _assert_parity(good_submission, task, runner=local_runner)
    record = _code_test_record(
        code_test_trace(good_submission, task), runner=local_runner
    )
    assert _derive_outcome(record).value == SubmissionOutcome.PASSED.value


def test_tests_failed_outcome_parity(
    task, failing_submission, local_runner
) -> None:
    _assert_parity(failing_submission, task, runner=local_runner)
    record = _code_test_record(
        code_test_trace(failing_submission, task), runner=local_runner
    )
    assert _derive_outcome(record).value == (
        SubmissionOutcome.TESTS_FAILED.value
    )


def test_no_top_level_functions_is_rejected_before_scoring(
    task, local_runner
) -> None:
    submission = "x = 1\n"  # compiles, no top-level functions
    scoring_outcome = _score_outcome(submission, task, runner=local_runner)
    assert scoring_outcome == SubmissionOutcome.PREPROCESSING_FAILED.value

    # Manually built code_test traces can still describe the lower-level
    # evaluator behavior outside official preprocessing acceptance.
    record = _code_test_record(
        code_test_trace(submission, task), runner=local_runner
    )
    assert _derive_outcome(record).value == (
        SubmissionOutcome.NO_TOP_LEVEL_FUNCTIONS.value
    )


def test_timed_out_outcome_parity(task, good_submission) -> None:
    runner = raising_runner(SubprocessTimeoutError("timed out"))
    _assert_parity(good_submission, task, runner=runner, timeout=1.0)
    record = _code_test_record(
        code_test_trace(good_submission, task), runner=runner, timeout=1.0
    )
    assert _derive_outcome(record).value == SubmissionOutcome.TIMED_OUT.value


def test_evaluation_incomplete_outcome_parity(task, good_submission) -> None:
    """Partial runner output (incomplete coverage, no failures) ⇒ incomplete."""

    def partial_runner(*, source, input_text, timeout_seconds):  # noqa: ANN001
        return SubprocessCompletedProcess(
            returncode=0,
            stdout='[{"case_id": "case_0", "status": "passed"}]',
            stderr="",
        )

    _assert_parity(good_submission, task, runner=partial_runner)
    record = _code_test_record(
        code_test_trace(good_submission, task), runner=partial_runner
    )
    assert _derive_outcome(record).value == (
        SubmissionOutcome.EVALUATION_INCOMPLETE.value
    )


# ---------------------------------------------------------------------------
# Facts stay in records; verdicts stay in the consumer.
# ---------------------------------------------------------------------------


def test_code_test_record_carries_no_verdict_fields(
    task, good_submission, local_runner
) -> None:
    """No thresholds, verdicts, or 'best'-as-judgement in records."""
    record = _code_test_record(
        code_test_trace(good_submission, task), runner=local_runner
    )
    assert "outcome" not in record.fact_values()
    assert "score" not in record.fact_values()
    assert "pass_at_k" not in record.fact_values()
    assert "passed" not in record.fact_values()


def test_derive_outcome_rejects_negative_counts() -> None:
    """A corrupt or tampered record cannot cancel failures with negative
    counts (failed_count=-1 + error_count=1 would otherwise read as zero
    failures and derive PASSED)."""
    from dr_code.eval import (
        Applicability,
        MetricFact,
        MetricRecord,
        OperatorCoordinates,
        OperatorLineage,
    )
    from dr_code.trace import (
        EXTERNAL_PRODUCER_ID,
        ExternalSource,
        TraceProducer,
    )

    operator = OperatorCoordinates(
        name="code_test",
        version="1",
        implementation_hash="c" * 64,
        settings=(
            ("task_key", "task"),
            ("timeout_seconds", 2.0),
        ),
    )
    question_identity = operator.question_identity_hash(on_key="input")
    lineage = OperatorLineage(
        evaluation_procedure_config_hash=_PROCEDURE_HASH,
        question_identity_hash=question_identity,
        operator="code_test",
        operator_version="1",
        operator_implementation="c" * 64,
    )
    record = MetricRecord.measured(
        question="code_test",
        question_identity_hash=question_identity,
        on_key="input",
        evaluation_procedure_config_hash=_PROCEDURE_HASH,
        trace_producer=TraceProducer(
            producer_id=EXTERNAL_PRODUCER_ID,
            external_source=ExternalSource(
                source_id="policy-example",
                content_digest="e" * 64,
            ),
        ),
        operator=operator,
        facts=tuple(
            MetricFact(
                name=name,
                value=value,
                unit=unit,
                applicability=Applicability.APPLICABLE,
                lineage=lineage,
            )
            for name, value, unit in (
                ("total_cases", 1, "case"),
                ("passed_count", 1, "case"),
                ("failed_count", -1, "case"),
                ("error_count", 1, "case"),
                ("timeout_count", 0, "case"),
                ("coverage_complete", True, "boolean"),
                ("function_count", 1, "function"),
                ("best_function_name", "candidate", "name"),
            )
        ),
    )
    with pytest.raises(ValueError, match="non-negative"):
        _derive_outcome(record)
