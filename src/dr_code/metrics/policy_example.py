"""Example caller policy derived from a neutral code-test record."""

from dr_code.humaneval.scoring import SubmissionOutcome
from dr_code.metrics.names import MetricName
from dr_code.metrics.records import MetricRecord, RecordStatus


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
