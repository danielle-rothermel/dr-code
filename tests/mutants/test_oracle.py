"""Execution-derived oracle + validation-gate tests.

Uses the real ``run_python_subprocess`` (fast, deterministic, local) for the
end-to-end oracle path and a fake runner for gate-logic edge cases.
"""

from __future__ import annotations

import pytest

from dr_code.mutants.oracle import (
    GateReport,
    InputOutcome,
    OracleError,
    OutcomeKind,
    ProgramOutcomes,
    distinct_input_indices,
    evaluate_gates,
    run_program_on_inputs,
)

_PROGRAM = "def f(a, b):\n    return a + b\n"


def test_run_program_captures_ordered_values() -> None:
    outcomes = run_program_on_inputs(
        program=_PROGRAM,
        entry_point="f",
        inputs=((1, 2), (10, 20), (0, 0)),
        timeout_seconds=5.0,
    )
    assert [o.output_repr for o in outcomes.outcomes] == ["3", "30", "0"]
    assert all(o.kind is OutcomeKind.VALUE for o in outcomes.outcomes)


def test_run_program_captures_exception_as_outcome() -> None:
    program = "def f(x):\n    return 1 / x\n"
    outcomes = run_program_on_inputs(
        program=program,
        entry_point="f",
        inputs=((2,), (0,)),
        timeout_seconds=5.0,
    )
    assert outcomes.outcomes[0] == InputOutcome(
        kind=OutcomeKind.VALUE, output_repr="0.5"
    )
    assert outcomes.outcomes[1] == InputOutcome(
        kind=OutcomeKind.ERROR, output_repr="ZeroDivisionError"
    )


def test_run_program_times_out_raises_oracle_error() -> None:
    program = "def f(x):\n    while True:\n        pass\n"
    with pytest.raises(OracleError):
        run_program_on_inputs(
            program=program,
            entry_point="f",
            inputs=((1,),),
            timeout_seconds=0.5,
        )


def test_distinct_indices_identifies_behavioral_difference() -> None:
    canonical = run_program_on_inputs(
        program="def f(a, b):\n    return a < b\n",
        entry_point="f",
        inputs=((1, 2), (2, 2), (3, 2)),
        timeout_seconds=5.0,
    )
    mutant = run_program_on_inputs(
        program="def f(a, b):\n    return a <= b\n",
        entry_point="f",
        inputs=((1, 2), (2, 2), (3, 2)),
        timeout_seconds=5.0,
    )
    # Only the tie input (2, 2) distinguishes < from <=.
    assert distinct_input_indices(canonical, mutant) == (1,)


def _outcomes(*values: str) -> ProgramOutcomes:
    return ProgramOutcomes(
        outcomes=tuple(
            InputOutcome(kind=OutcomeKind.VALUE, output_repr=v)
            for v in values
        )
    )


def test_gates_accept_distinct_deterministic_mutant() -> None:
    report = evaluate_gates(
        canonical=_outcomes("1", "2", "3"),
        mutant_first=_outcomes("1", "9", "3"),
        mutant_second=_outcomes("1", "9", "3"),
    )
    assert report.accepted
    assert report.distinct_input_count == 1
    assert report.rejection_reason() is None


def test_gates_reject_non_distinct_mutant() -> None:
    report = evaluate_gates(
        canonical=_outcomes("1", "2", "3"),
        mutant_first=_outcomes("1", "2", "3"),
        mutant_second=_outcomes("1", "2", "3"),
    )
    assert not report.accepted
    assert report.rejection_reason() == (
        "behaviorally identical to canonical on every input"
    )


def test_gates_reject_nondeterministic_mutant() -> None:
    report = evaluate_gates(
        canonical=_outcomes("1", "2", "3"),
        mutant_first=_outcomes("1", "9", "3"),
        mutant_second=_outcomes("1", "8", "3"),
    )
    assert not report.accepted
    assert report.rejection_reason() == "non-deterministic across two runs"


def test_gate_report_flag_precedence() -> None:
    # Non-termination is reported before distinctness.
    report = GateReport(
        terminates=False,
        deterministic=True,
        distinct_input_count=5,
        serializable=True,
    )
    assert not report.accepted
    assert "terminate" in (report.rejection_reason() or "")
