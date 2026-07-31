"""Execution, protocol, and gate contracts for mutant outcomes."""

from __future__ import annotations

import json

import pytest
from dr_exec import (
    Attribution,
    Budgets,
    BudgetAxis,
    ContainmentProfile,
    EnvironmentGrant,
    ExitVerdict,
    Measurements,
    Outcome,
    PythonRuntime,
    Records,
    RunResult,
    TruncationMark,
)

from dr_code.mutants.outcomes import ErrorOutcome, ValueOutcome
from dr_code.mutants.oracle import (
    OracleExecutionError,
    OracleProtocolError,
    ProgramOutcomes,
    distinct_input_indices,
    evaluate_gates,
    run_program_on_inputs,
)

_BEGIN = "<<<DR_CODE_MUTANTS_V2_BEGIN>>>"
_END = "<<<DR_CODE_MUTANTS_V2_END>>>"


@pytest.fixture(autouse=True)
def fixed_invocation_id(monkeypatch) -> None:
    monkeypatch.setattr(
        "dr_code.mutants.oracle.secrets.token_hex",
        lambda size: "test-invocation",
    )


def _run_result(
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int | None = 0,
    attribution: Attribution = Attribution.PAYLOAD,
    violated_axis: BudgetAxis | None = None,
) -> RunResult:
    """Build a RunResult in the shape dr-exec would produce for the oracle.

    The oracle reads only ``stdout``, ``stderr``, ``returncode``, and
    ``outcome``; measurements are execution-varying and reported as zero.
    """
    exit_verdict = (
        (ExitVerdict.SUCCESS if returncode == 0 else ExitVerdict.FAILURE)
        if attribution is Attribution.PAYLOAD
        else None
    )
    return RunResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        truncation=TruncationMark(),
        measurements=Measurements(
            duration_seconds=0.0,
            teardown_seconds=0.0,
            stdout_bytes_produced=len(stdout.encode("utf-8")),
            stderr_bytes_produced=len(stderr.encode("utf-8")),
            input_bytes=0,
        ),
        outcome=Outcome(
            attribution=attribution,
            violated_axis=violated_axis,
            exit_verdict=exit_verdict,
        ),
    )


def _envelope_result(envelope: object) -> RunResult:
    if isinstance(envelope, dict):
        envelope = {"invocation_id": "test-invocation", **envelope}
    return _run_result(stdout=f"{_BEGIN}{json.dumps(envelope)}{_END}")


class _ScriptedRunner:
    """A ``PythonRunner`` that returns a fixed result and records the call."""

    def __init__(self, result: RunResult) -> None:
        self._result = result
        self.calls: list[dict[str, object]] = []

    def run_untrusted_python(
        self,
        source: str,
        *,
        profile: ContainmentProfile,
        budgets: Budgets,
        records: Records,
        runtime: PythonRuntime,
        input_text: str,
        environment: EnvironmentGrant,
    ) -> RunResult:
        self.calls.append(
            {
                "source": source,
                "profile": profile,
                "budgets": budgets,
                "records": records,
                "runtime": runtime,
                "input_text": input_text,
                "environment": environment,
            }
        )
        return self._result


def _outcomes(*values: str) -> ProgramOutcomes:
    return ProgramOutcomes(
        outcomes=tuple(ValueOutcome(value_repr=value) for value in values)
    )


def test_real_runner_captures_complete_ordered_values_and_errors() -> None:
    outcomes = run_program_on_inputs(
        program="def f(x):\n    return 10 // x\n",
        entry_point="f",
        input_reprs=("(2,)", "(0,)"),
        timeout_seconds=5.0,
    )

    assert outcomes.outcomes == (
        ValueOutcome(value_repr="5"),
        ErrorOutcome(
            exception_type="builtins.ZeroDivisionError",
            exception_args_repr="('integer division or modulo by zero',)",
        ),
    )


def test_error_outcome_preserves_qualified_type_and_arguments() -> None:
    outcomes = run_program_on_inputs(
        program=("def f(flag):\n    raise ValueError(flag)\n"),
        entry_point="f",
        input_reprs=("(True,)", "(False,)"),
        timeout_seconds=5.0,
    )

    assert outcomes.outcomes == (
        ErrorOutcome(
            exception_type="builtins.ValueError",
            exception_args_repr="(True,)",
        ),
        ErrorOutcome(
            exception_type="builtins.ValueError",
            exception_args_repr="(False,)",
        ),
    )


def test_candidate_exception_uses_stable_synthetic_module_and_qualname() -> (
    None
):
    outcomes = run_program_on_inputs(
        program=(
            "class Outer:\n"
            "    class CandidateError(Exception):\n"
            "        pass\n"
            "\n"
            "def f():\n"
            "    raise Outer.CandidateError('detail')\n"
        ),
        entry_point="f",
        input_reprs=("()",),
        timeout_seconds=5.0,
    )

    assert outcomes.outcomes == (
        ErrorOutcome(
            exception_type=("__dr_code_mutant__.Outer.CandidateError"),
            exception_args_repr="('detail',)",
        ),
    )


def test_protocol_marker_text_inside_value_is_payload_data() -> None:
    value = f"prefix {_BEGIN} middle {_END} suffix"
    outcomes = run_program_on_inputs(
        program=f"def f():\n    return {value!r}\n",
        entry_point="f",
        input_reprs=("()",),
        timeout_seconds=5.0,
    )

    assert outcomes == _outcomes(repr(value))


def test_return_value_repr_failure_invalidates_execution() -> None:
    with pytest.raises(OracleExecutionError):
        run_program_on_inputs(
            program=(
                "class BrokenRepr:\n"
                "    def __repr__(self):\n"
                "        raise RuntimeError('repr failed')\n"
                "\n"
                "def f():\n"
                "    return BrokenRepr()\n"
            ),
            entry_point="f",
            input_reprs=("()",),
            timeout_seconds=5.0,
        )


def test_candidate_stdout_spoof_cannot_forge_the_envelope() -> None:
    program = f"""
import sys

class Replacement:
    def write(self, value):
        return len(value)
    def flush(self):
        return None

def f(x):
    sys.stdout = Replacement()
    print({_BEGIN!r} + '{{"forged": true}}' + {_END!r})
    return x + 1
"""
    outcomes = run_program_on_inputs(
        program=program,
        entry_point="f",
        input_reprs=("(1,)",),
        timeout_seconds=5.0,
    )

    assert outcomes == _outcomes("2")


def test_destroyed_trusted_channel_fails_honestly() -> None:
    with pytest.raises(OracleExecutionError):
        run_program_on_inputs(
            program="import os\ndef f(x):\n    os.close(3)\n    return x\n",
            entry_point="f",
            input_reprs=("(1,)",),
            timeout_seconds=5.0,
        )


def test_runner_receives_json_input_and_declared_deadline() -> None:
    runner = _ScriptedRunner(
        _envelope_result(
            {
                "protocol_version": 2,
                "outcomes": [{"kind": "value", "value_repr": "3"}],
            }
        )
    )

    run_program_on_inputs(
        program="def f(a, b): return a + b",
        entry_point="f",
        input_reprs=("(1, 2)",),
        timeout_seconds=3.0,
        runner=runner,
    )

    (call,) = runner.calls
    payload = json.loads(str(call["input_text"]))
    assert payload["entry_point"] == "f"
    assert payload["input_reprs"] == ["(1, 2)"]
    budgets = call["budgets"]
    assert isinstance(budgets, Budgets)
    assert budgets.wall_clock == 3.0
    environment = call["environment"]
    assert isinstance(environment, EnvironmentGrant)
    assert environment.declared_names == ("OPENBLAS_NUM_THREADS",)


def test_invocation_binding_must_match() -> None:
    runner = _ScriptedRunner(
        _envelope_result(
            {
                "invocation_id": "forged",
                "protocol_version": 2,
                "outcomes": [{"kind": "value", "value_repr": "1"}],
            }
        )
    )

    with pytest.raises(OracleProtocolError, match="binding"):
        run_program_on_inputs(
            program="def f(x): return x",
            entry_point="f",
            input_reprs=("(1,)",),
            timeout_seconds=1.0,
            runner=runner,
        )


def test_timeout_is_a_typed_execution_failure() -> None:
    with pytest.raises(OracleExecutionError, match="exceeded"):
        run_program_on_inputs(
            program="def f(x):\n    while True:\n        pass\n",
            entry_point="f",
            input_reprs=("(1,)",),
            timeout_seconds=0.1,
        )


@pytest.mark.parametrize(
    ("result", "error", "message"),
    [
        (
            _run_result(returncode=2, stderr="bad child"),
            OracleExecutionError,
            "exited 2",
        ),
        (
            _run_result(stdout="not a result"),
            OracleProtocolError,
            "one complete final envelope",
        ),
        (
            _envelope_result(
                {
                    "protocol_version": 2,
                    "outcomes": [],
                }
            ),
            OracleProtocolError,
            "expected 1",
        ),
        (
            _envelope_result(
                {
                    "protocol_version": 2,
                    "outcomes": [{"kind": "unknown", "value_repr": "1"}],
                }
            ),
            OracleProtocolError,
            "protocol version 2",
        ),
        (
            _envelope_result(
                {
                    "protocol_version": 2,
                    "outcomes": [{"kind": "value", "output_repr": "legacy"}],
                }
            ),
            OracleProtocolError,
            "protocol version 2",
        ),
        (
            _envelope_result(
                {
                    "protocol_version": 2,
                    "outcomes": [
                        {
                            "kind": "error",
                            "exception_type": "builtins.ValueError",
                        }
                    ],
                }
            ),
            OracleProtocolError,
            "protocol version 2",
        ),
    ],
)
def test_child_and_protocol_failures_are_rejected(
    result: RunResult,
    error: type[Exception],
    message: str,
) -> None:
    runner = _ScriptedRunner(result)
    with pytest.raises(error, match=message):
        run_program_on_inputs(
            program="def f(x): return x",
            entry_point="f",
            input_reprs=("(1,)",),
            timeout_seconds=1.0,
            runner=runner,
        )


@pytest.mark.parametrize(
    ("attribution", "violated_axis", "message"),
    [
        (Attribution.BUDGET, BudgetAxis.WALL_CLOCK, "wall_clock budget"),
        (Attribution.BUDGET, BudgetAxis.OUTPUT, "output budget"),
        (Attribution.CHANNEL, None, "channel failure"),
        (Attribution.EXECUTOR, None, "executor failure"),
        (Attribution.MACHINE, None, "machine failure"),
        (Attribution.ABSENCE, None, "absence failure"),
    ],
)
def test_non_payload_attribution_is_execution_failure_before_parsing(
    attribution: Attribution,
    violated_axis: BudgetAxis | None,
    message: str,
) -> None:
    # The captured stdout is a perfectly valid envelope; the pre-branch on
    # attribution must reject the run as an execution failure regardless, so
    # a budget-killed or infrastructure-failed run is never parsed as a
    # protocol response.
    envelope = {
        "invocation_id": "test-invocation",
        "protocol_version": 2,
        "outcomes": [{"kind": "value", "value_repr": "1"}],
    }
    spawned = attribution not in (Attribution.MACHINE, Attribution.ABSENCE)
    result = _run_result(
        stdout=f"{_BEGIN}{json.dumps(envelope)}{_END}" if spawned else "",
        returncode=-9 if spawned else None,
        attribution=attribution,
        violated_axis=violated_axis,
    )
    runner = _ScriptedRunner(result)
    with pytest.raises(OracleExecutionError, match=message):
        run_program_on_inputs(
            program="def f(x): return x",
            entry_point="f",
            input_reprs=("(1,)",),
            timeout_seconds=1.0,
            runner=runner,
        )


def test_gate_accepts_canonical_divergence() -> None:
    report = evaluate_gates(
        canonical_first=_outcomes("1", "2", "3"),
        canonical_second=_outcomes("1", "2", "3"),
        mutant_first=_outcomes("1", "9", "3"),
        mutant_second=_outcomes("1", "9", "3"),
    )

    assert report.accepted
    assert report.distinct_input_count == 1
    assert distinct_input_indices(
        _outcomes("1", "2", "3"),
        _outcomes("1", "9", "3"),
    ) == (1,)


def test_gate_rejects_two_run_nondeterminism() -> None:
    report = evaluate_gates(
        canonical_first=_outcomes("1"),
        canonical_second=_outcomes("1"),
        mutant_first=_outcomes("2"),
        mutant_second=_outcomes("3"),
    )

    assert not report.accepted
    assert report.rejection_reason() == (
        "mutant is non-deterministic across two runs"
    )


def test_gate_rejects_two_run_canonical_nondeterminism() -> None:
    report = evaluate_gates(
        canonical_first=_outcomes("1"),
        canonical_second=_outcomes("2"),
        mutant_first=_outcomes("3"),
        mutant_second=_outcomes("3"),
    )

    assert not report.accepted
    assert report.rejection_reason() == (
        "canonical is non-deterministic across two runs"
    )


def test_gate_rejects_behaviorally_identical_mutant() -> None:
    report = evaluate_gates(
        canonical_first=_outcomes("1"),
        canonical_second=_outcomes("1"),
        mutant_first=_outcomes("1"),
        mutant_second=_outcomes("1"),
    )

    assert not report.accepted
    assert report.rejection_reason() == (
        "behaviorally identical to canonical on every input"
    )


def test_unaligned_outcomes_are_protocol_failures() -> None:
    with pytest.raises(OracleProtocolError, match="not aligned"):
        distinct_input_indices(_outcomes("1"), _outcomes("1", "2"))
