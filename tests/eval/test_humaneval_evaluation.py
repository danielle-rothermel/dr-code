"""Crosswalk: operational HumanEval results map onto kernel facts/records.

These pin the step-3 seam: canonical candidate-execution identity from the
four required coordinates (seed-invariant), lossless CodeTestResult -> facts
conversion, and the explicit empty-candidate-set (not failure) record.
"""

from __future__ import annotations

from dr_code.eval.facts import (
    AbsenceMode,
    Applicability,
    RecordStatus,
)
from dr_code.eval.humaneval_evaluation import (
    CODE_TEST_OPERATOR,
    candidate_execution_identity,
    code_test_facts,
    code_test_record,
    compile_facts_for_candidate,
    empty_candidate_set_record,
    evaluation_procedure_config_identity,
    kernel_metric_extraction_definition,
    record_from_result_row,
)
from dr_code.metrics.definition import MetricQuestion, MetricsDefinition
from dr_code.metrics.names import MetricName
from dr_code.metrics.operators.code_test import CodeTestResult
from dr_code.preprocessing.definitions import (
    HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
)


def _metrics_definition() -> MetricsDefinition:
    return MetricsDefinition(
        definition_id="humaneval-metrics",
        version="v1",
        questions=(
            MetricQuestion(
                metric=MetricName.CODE_TEST,
                on="code",
                settings={"task_key": "task", "timeout_seconds": 2.0},
            ),
        ),
    )

_CFG = "epc-hash-1"


def _result() -> CodeTestResult:
    return CodeTestResult(
        total_cases=5,
        passed_count=5,
        failed_count=0,
        error_count=0,
        timeout_count=0,
        coverage_complete=True,
        function_count=1,
        best_function_name="solve",
    )


def test_candidate_execution_identity_is_deterministic() -> None:
    a = candidate_execution_identity(
        task_identity="t1",
        candidate_source="def f():\n    return 1\n",
        evaluation_procedure_config_hash=_CFG,
        execution_fingerprint="fp1",
    )
    b = candidate_execution_identity(
        task_identity="t1",
        candidate_source="def f():\n    return 1\n",
        evaluation_procedure_config_hash=_CFG,
        execution_fingerprint="fp1",
    )
    assert a == b
    assert len(a) == 64  # full lowercase SHA-256 identity hash


def test_candidate_execution_identity_depends_on_each_coordinate() -> None:
    base = dict(
        task_identity="t1",
        candidate_source="def f():\n    return 1\n",
        evaluation_procedure_config_hash=_CFG,
        execution_fingerprint="fp1",
    )
    baseline = candidate_execution_identity(**base)
    assert candidate_execution_identity(**{**base, "task_identity": "t2"}) != (
        baseline
    )
    assert candidate_execution_identity(
        **{**base, "candidate_source": "def f():\n    return 2\n"}
    ) != baseline
    assert candidate_execution_identity(
        **{**base, "evaluation_procedure_config_hash": "epc-2"}
    ) != baseline
    assert candidate_execution_identity(
        **{**base, "execution_fingerprint": "fp2"}
    ) != baseline


def test_execution_identity_excludes_rng_seed_by_construction() -> None:
    # The identity has no seed input at all; two runs differing only in the
    # seed used to sample the candidate share one execution identity.
    same = candidate_execution_identity(
        task_identity="t1",
        candidate_source="def f():\n    return 1\n",
        evaluation_procedure_config_hash=_CFG,
        execution_fingerprint="fp1",
    )
    again = candidate_execution_identity(
        task_identity="t1",
        candidate_source="def f():\n    return 1\n",
        evaluation_procedure_config_hash=_CFG,
        execution_fingerprint="fp1",
    )
    assert same == again


def test_code_test_facts_are_lossless_and_carry_units() -> None:
    facts = code_test_facts(
        _result(), evaluation_procedure_config_hash=_CFG
    )
    by_name = {f.name: f for f in facts}
    # Every descriptive field survives as a fact.
    assert {
        "total_cases",
        "passed_count",
        "failed_count",
        "error_count",
        "timeout_count",
        "coverage_complete",
        "function_count",
        "best_function_name",
    } <= set(by_name)
    assert by_name["total_cases"].value == 5
    assert by_name["total_cases"].unit == "case"
    assert by_name["coverage_complete"].unit == "boolean"
    assert by_name["coverage_complete"].value is True
    assert by_name["best_function_name"].value == "solve"
    # All facts carry resolved operator lineage.
    for fact in facts:
        assert fact.applicability is Applicability.APPLICABLE
        assert fact.lineage.operator == CODE_TEST_OPERATOR
        assert fact.lineage.evaluation_procedure_config_hash == _CFG
        assert fact.lineage.operator_version  # resolved, non-empty


def test_absent_best_function_name_drops_only_that_fact() -> None:
    result = _result().model_copy(update={"best_function_name": None})
    names = {
        f.name
        for f in code_test_facts(
            result, evaluation_procedure_config_hash=_CFG
        )
    }
    assert "best_function_name" not in names
    assert "total_cases" in names  # the rest are still lossless


def test_code_test_record_is_measured() -> None:
    record = code_test_record(
        _result(), evaluation_procedure_config_hash=_CFG
    )
    assert record.status is RecordStatus.MEASURED
    assert record.evaluation_procedure_config_hash == _CFG
    assert record.facts  # measured records carry facts
    assert record.absence_mode is None


def test_empty_candidate_set_is_not_a_preprocessing_failure() -> None:
    record = empty_candidate_set_record(
        evaluation_procedure_config_hash=_CFG
    )
    assert record.status is RecordStatus.NOT_APPLICABLE
    assert record.absence_mode is AbsenceMode.EMPTY_CANDIDATE_SET
    # Explicitly distinct from the causal preprocessing-failure absence.
    assert record.absence_mode is not AbsenceMode.PREPROCESSING_FAILURE
    assert record.absence_cause


def test_compile_gate_bridges_kernel_code_artifact() -> None:
    assert compile_facts_for_candidate("def f():\n    return 1\n") is True
    assert compile_facts_for_candidate("def f(:\n") is False


def test_kernel_preprocessing_definition_is_lossless_and_canonical() -> None:
    from dr_code.eval.humaneval_evaluation import (
        kernel_preprocessing_definition,
    )
    from dr_code.preprocessing.definition import (
        PreprocessingDefinition as OpDef,
        StepSpec,
    )

    op = OpDef(
        definition_id="d1",
        version="1",
        steps=(
            StepSpec(
                instance_name="extract",
                step="extract_candidates",
                settings={"b": 2, "a": 1},
            ),
            StepSpec(instance_name="dedupe", step="dedupe_candidates"),
        ),
    )
    kernel = kernel_preprocessing_definition(op)
    # Every step instance survives, in order, with settings preserved.
    assert kernel.definition_id == "d1"
    assert kernel.version == "1"
    assert [b.instance_name for b in kernel.steps] == ["extract", "dedupe"]
    assert dict(kernel.steps[0].settings) == {"a": 1, "b": 2}
    # Canonical eval-kernel identity (full SHA-256), deterministic.
    identity = kernel.identity_hash()
    assert len(identity) == 64
    assert identity == kernel_preprocessing_definition(op).identity_hash()


def test_metric_extraction_crosswalk_is_lossless_and_canonical() -> None:
    kernel = kernel_metric_extraction_definition(_metrics_definition())
    assert kernel.definition_id == "humaneval-metrics"
    assert len(kernel.questions) == 1
    q = kernel.questions[0]
    assert q.metric == str(MetricName.CODE_TEST)
    assert q.on == "code"
    assert dict(q.settings) == {"task_key": "task", "timeout_seconds": 2.0}
    identity = kernel.materialize().config_identity_hash
    assert len(identity) == 64


def test_procedure_config_identity_folds_both_components() -> None:
    identity = evaluation_procedure_config_identity(
        preprocessing=HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
        metrics=_metrics_definition(),
    )
    assert len(identity) == 64
    # Deterministic across calls.
    assert identity == evaluation_procedure_config_identity(
        preprocessing=HUMANEVAL_FUNCTION_CANDIDATES_V1_DEFINITION,
        metrics=_metrics_definition(),
    )


def test_record_from_measured_row_reconstructs_facts() -> None:
    row = {
        "record_status": "measured",
        "total_cases": 5,
        "passed_count": 5,
        "failed_count": 0,
        "error_count": 0,
        "timeout_count": 0,
        "coverage_complete": True,
        "function_count": 1,
        "best_function_name": "solve",
    }
    record = record_from_result_row(
        row, evaluation_procedure_config_hash="epc"
    )
    assert record.status is RecordStatus.MEASURED
    names = {f.name for f in record.facts}
    assert "passed_count" in names and "total_cases" in names


def test_record_from_incomplete_row_is_operator_failure_not_success() -> None:
    row = {
        "record_status": "evaluation_incomplete",
        "failure_type": "infrastructure",
        "failure_message": "worker died",
    }
    record = record_from_result_row(
        row, evaluation_procedure_config_hash="epc"
    )
    assert record.status is RecordStatus.OPERATOR_FAILURE
    assert record.failure_type == "infrastructure"
    assert not record.facts  # never a silent success


def test_pass_rate_reconciles_with_descriptive_count_rate() -> None:
    from dr_code.eval.aggregation import AggregationStatus
    from dr_code.eval.humaneval_evaluation import pass_rate

    indicators = (True, True, False, True, False)
    output = pass_rate(indicators)
    assert output.status is AggregationStatus.OK
    # Kernel mean reduction reconciles with the naive passed/total rate.
    passed = sum(1 for i in indicators if i)
    assert output.value == passed / len(indicators)
    assert output.count_present == len(indicators)


def test_pass_rate_missing_outcome_propagates_not_silent_failure() -> None:
    from dr_code.eval.aggregation import AggregationStatus
    from dr_code.eval.humaneval_evaluation import pass_rate

    # A missing outcome must not be silently counted as a failure.
    output = pass_rate((True, None, True))
    assert output.status is AggregationStatus.MISSING_DATA
    assert output.value is None


def test_pass_rate_empty_denominator_is_not_applicable() -> None:
    from dr_code.eval.aggregation import AggregationStatus
    from dr_code.eval.humaneval_evaluation import pass_rate

    output = pass_rate(())
    assert output.status is AggregationStatus.NOT_APPLICABLE
    assert output.value is None
