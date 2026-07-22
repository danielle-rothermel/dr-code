"""Metric Facts, Records, Scores, and distinguishable absence modes."""

from __future__ import annotations

import pytest

from dr_code.eval.facts import (
    AbsenceMode,
    Applicability,
    MetricFact,
    MetricRecord,
    OperatorLineage,
    Score,
)


def _lineage() -> OperatorLineage:
    return OperatorLineage(
        evaluation_procedure_config_hash="proc-hash",
        operator="code_leakage",
        operator_version="2",
    )


def _fact() -> MetricFact:
    return MetricFact(
        name="leaked_lines",
        value=3,
        unit="lines",
        applicability=Applicability.APPLICABLE,
        lineage=_lineage(),
    )


def test_fact_requires_explicit_unit() -> None:
    with pytest.raises(ValueError, match="explicit unit"):
        MetricFact(
            name="x",
            value=1,
            unit="",
            applicability=Applicability.APPLICABLE,
            lineage=_lineage(),
        )


def test_fact_carries_operator_step_lineage() -> None:
    lineage = OperatorLineage(
        evaluation_procedure_config_hash="proc-hash",
        operator="select_first_metric",
        operator_version="1",
        step="select_first",
        step_version="1",
    )
    fact = MetricFact(
        name="n",
        value=1,
        unit="count",
        applicability=Applicability.APPLICABLE,
        lineage=lineage,
    )
    assert fact.lineage.step == "select_first"
    assert fact.lineage.step_version == "1"


def test_measured_record_requires_facts() -> None:
    with pytest.raises(ValueError, match="at least one fact"):
        MetricRecord.measured(
            question="q",
            on_key="output",
            evaluation_procedure_config_hash="proc",
            facts=(),
        )


def test_record_shapes_are_mutually_exclusive() -> None:
    record = MetricRecord.measured(
        question="q",
        on_key="output",
        evaluation_procedure_config_hash="proc",
        facts=(_fact(),),
    )
    assert record.absence_mode is None
    assert record.failure_type is None


def test_not_applicable_record_carries_absence_mode() -> None:
    record = MetricRecord.not_applicable(
        question="q",
        on_key="output",
        evaluation_procedure_config_hash="proc",
        absence_mode=AbsenceMode.PREPROCESSING_FAILURE,
        cause="select_first produced no candidate",
    )
    assert record.absence_mode is AbsenceMode.PREPROCESSING_FAILURE
    assert record.facts == ()


def test_operator_failure_record_requires_type_and_message() -> None:
    with pytest.raises(ValueError, match="failure type and message"):
        MetricRecord.operator_failure(
            question="q",
            on_key="output",
            evaluation_procedure_config_hash="proc",
            failure_type="",
            failure_message="",
        )


def test_absence_modes_are_all_distinguishable() -> None:
    """Native Absent (Preprocessing Failure on present input) is distinct
    from no-input, no-trace, missing-key, and empty-candidate-set."""

    modes = [
        AbsenceMode.PREPROCESSING_FAILURE,
        AbsenceMode.NO_INPUT,
        AbsenceMode.NO_TRACE,
        AbsenceMode.MISSING_TRACE_KEY,
        AbsenceMode.EMPTY_CANDIDATE_SET,
    ]
    records = [
        MetricRecord.not_applicable(
            question="q",
            on_key="output",
            evaluation_procedure_config_hash="proc",
            absence_mode=mode,
            cause=f"cause for {mode.value}",
        )
        for mode in modes
    ]
    seen = {record.absence_mode for record in records}
    assert len(seen) == len(modes)
    # Only PREPROCESSING_FAILURE is the native Absent / Preprocessing
    # Failure role; the others are explicitly not that.
    non_failure = [
        record
        for record in records
        if record.absence_mode is not AbsenceMode.PREPROCESSING_FAILURE
    ]
    assert len(non_failure) == 4


def test_score_derives_from_facts_and_retains_lineage() -> None:
    score = Score(
        name="leakage_ratio",
        value=0.5,
        unit="ratio",
        evaluation_procedure_config_hash="proc",
        derived_from=("leaked_lines", "total_lines"),
    )
    assert score.derived_from == ("leaked_lines", "total_lines")
    assert score.evaluation_procedure_config_hash == "proc"


def test_score_requires_explicit_unit() -> None:
    with pytest.raises(ValueError, match="explicit unit"):
        Score(
            name="s",
            value=1.0,
            unit="",
            evaluation_procedure_config_hash="proc",
            derived_from=("x",),
        )
