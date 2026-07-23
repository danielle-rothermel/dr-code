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
)
from dr_code.metrics.operators.code_test import CodeTestResult

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
