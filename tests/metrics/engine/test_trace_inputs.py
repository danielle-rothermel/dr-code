"""Trace-input binding contracts for the metrics engine."""

from __future__ import annotations

import pytest

from dr_code.trace import (
    Absent,
    CodeArtifact,
    TextArtifact,
    WiringError,
    external_trace,
)

from ._helpers import _definition, _extract, _extract_batch, _q


def test_missing_on_key_is_a_wiring_error_before_any_work(
    counting_runner,
) -> None:
    trace = external_trace(
        {"input": TextArtifact(text="hi"), "output": TextArtifact(text="hi")}
    )
    definition = _definition([_q("text_stats", on="nonexistent")])
    with pytest.raises(WiringError):
        _extract(definition, trace, run_in_sandbox=counting_runner)
    assert counting_runner.call_count == 0


def test_wrong_artifact_kind_is_a_wiring_error(counting_runner) -> None:
    """ast_stats requires CODE; a TEXT key is a kind mismatch (bind-time)."""
    trace = external_trace(
        {
            "input": TextArtifact(text="not code"),
            "output": TextArtifact(text="x"),
        }
    )
    definition = _definition([_q("ast_stats", on="input")])
    with pytest.raises(WiringError):
        _extract(definition, trace, run_in_sandbox=counting_runner)
    assert counting_runner.call_count == 0


def test_missing_auxiliary_key_is_a_wiring_error(counting_runner) -> None:
    """code_test needs its task key; a key absent from the namespace is wiring."""
    candidate = "def add_one(x):\n    return x + 1\n"
    trace = external_trace(
        {
            "input": CodeArtifact(source=candidate),
            "output": CodeArtifact(source=candidate),
        }
    )
    definition = _definition([_q("code_test", on="input")])
    with pytest.raises(WiringError):
        _extract(definition, trace, run_in_sandbox=counting_runner)
    assert counting_runner.call_count == 0


def test_batch_wiring_error_runs_no_sandbox_work(counting_runner) -> None:
    bad = external_trace(
        {
            "input": TextArtifact(text="not code"),
            "output": TextArtifact(text="x"),
        }
    )
    definition = _definition([_q("ast_stats", on="input")])
    with pytest.raises(WiringError):
        _extract_batch(definition, [bad, bad], run_in_sandbox=counting_runner)
    assert counting_runner.call_count == 0


def test_absent_on_key_yields_not_applicable_with_cause() -> None:
    """Absent input ⇒ NOT_APPLICABLE carrying the Absent lineage,
    distinct from a missing key (which is a wiring error)."""
    trace = external_trace(
        {
            "input": Absent(
                failed_step="extract",
                failure_code="no_alternative_produced_candidates",
                cause="no code",
            ),
            "output": Absent(
                failed_step="extract",
                failure_code="no_alternative_produced_candidates",
                cause="no code",
            ),
        }
    )
    definition = _definition(
        [_q("text_stats", on="input"), _q("ast_stats", on="input")]
    )
    records = _extract(definition, trace)
    assert len(records) == 2
    for record in records:
        assert record.status.value == "not_applicable"
        assert record.absence.failed_step == "extract"
        assert record.absence.failure_code == (
            "no_alternative_produced_candidates"
        )
        assert record.absence.cause == "no code"


def test_absent_auxiliary_yields_not_applicable() -> None:
    """code_test whose task aux is present-but-Absent is not-applicable, not a
    wiring bug."""
    candidate = "def add_one(x):\n    return x + 1\n"
    code = CodeArtifact(source=candidate)
    trace = external_trace(
        {
            "input": code,
            "output": code,
            "task": Absent(
                failed_step="load",
                failure_code="missing_task",
                cause="missing task",
            ),
        }
    )
    definition = _definition([_q("code_test", on="input")])
    record = _extract(definition, trace)[0]
    assert record.status.value == "not_applicable"
    assert record.absence.failed_step == "load"
    assert record.absence.failure_code == "missing_task"
