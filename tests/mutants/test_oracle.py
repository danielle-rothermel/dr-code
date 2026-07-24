"""Execution, protocol, and gate contracts for mutant outcomes."""

from __future__ import annotations

import json

import pytest

from dr_code.execution.subprocess import (
    SubprocessCompletedProcess,
    SubprocessTimeoutError,
)
from dr_code.mutants import oracle as oracle_module
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


def _completed(envelope: object) -> SubprocessCompletedProcess:
    if isinstance(envelope, dict):
        envelope = {"invocation_id": "test-invocation", **envelope}
    return SubprocessCompletedProcess(
        returncode=0,
        stdout=f"{_BEGIN}{json.dumps(envelope)}{_END}",
        stderr="",
    )


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


def test_runner_receives_json_as_input_text() -> None:
    captured: dict[str, object] = {}

    def runner(
        *,
        source: str,
        input_text: str,
        timeout_seconds: float,
    ) -> SubprocessCompletedProcess:
        captured.update(
            source=source,
            input_text=input_text,
            timeout_seconds=timeout_seconds,
        )
        return _completed(
            {
                "protocol_version": 2,
                "outcomes": [{"kind": "value", "value_repr": "3"}],
            }
        )

    run_program_on_inputs(
        program="def f(a, b): return a + b",
        entry_point="f",
        input_reprs=("(1, 2)",),
        timeout_seconds=3.0,
        runner=runner,
    )

    payload = json.loads(str(captured["input_text"]))
    assert payload["entry_point"] == "f"
    assert payload["input_reprs"] == ["(1, 2)"]
    assert captured["timeout_seconds"] == 3.0


def test_invocation_binding_must_match() -> None:
    def runner(
        *,
        source: str,
        input_text: str,
        timeout_seconds: float,
    ) -> SubprocessCompletedProcess:
        _ = source, input_text, timeout_seconds
        return _completed(
            {
                "invocation_id": "forged",
                "protocol_version": 2,
                "outcomes": [{"kind": "value", "value_repr": "1"}],
            }
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
    program = "def f(x): return x"
    input_reprs = ("(1,)",)
    timeout_seconds = 3.25
    timeout = SubprocessTimeoutError(
        f"subprocess exceeded {timeout_seconds} seconds"
    )
    requests: list[dict[str, object]] = []

    def runner(
        *,
        source: str,
        input_text: str,
        timeout_seconds: float,
    ) -> SubprocessCompletedProcess:
        requests.append(
            {
                "source": source,
                "input_text": input_text,
                "timeout_seconds": timeout_seconds,
            }
        )
        raise timeout

    with pytest.raises(
        OracleExecutionError,
        match=r"^subprocess exceeded 3\.25 seconds$",
    ) as raised:
        run_program_on_inputs(
            program=program,
            entry_point="f",
            input_reprs=input_reprs,
            timeout_seconds=timeout_seconds,
            runner=runner,
        )

    assert requests == [
        {
            "source": oracle_module._RUNNER_SOURCE,
            "input_text": json.dumps(
                {
                    "entry_point": "f",
                    "input_reprs": input_reprs,
                    "invocation_id": "test-invocation",
                    "program": program,
                },
                sort_keys=True,
            ),
            "timeout_seconds": timeout_seconds,
        }
    ]
    assert raised.value.__cause__ is timeout


@pytest.mark.parametrize(
    ("completed", "message"),
    [
        (
            SubprocessCompletedProcess(
                returncode=2,
                stdout="",
                stderr="bad child",
            ),
            "exited 2",
        ),
        (
            SubprocessCompletedProcess(
                returncode=0,
                stdout="not a result",
                stderr="",
            ),
            "one complete final envelope",
        ),
        (
            _completed(
                {
                    "protocol_version": 2,
                    "outcomes": [],
                }
            ),
            "expected 1",
        ),
        (
            _completed(
                {
                    "protocol_version": 2,
                    "outcomes": [{"kind": "unknown", "value_repr": "1"}],
                }
            ),
            "protocol version 2",
        ),
        (
            _completed(
                {
                    "protocol_version": 2,
                    "outcomes": [{"kind": "value", "output_repr": "legacy"}],
                }
            ),
            "protocol version 2",
        ),
        (
            _completed(
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
            "protocol version 2",
        ),
    ],
)
def test_child_and_protocol_failures_are_rejected(
    completed: SubprocessCompletedProcess,
    message: str,
) -> None:
    def runner(
        *,
        source: str,
        input_text: str,
        timeout_seconds: float,
    ) -> SubprocessCompletedProcess:
        _ = source, input_text, timeout_seconds
        return completed

    error = (
        OracleExecutionError if completed.returncode else OracleProtocolError
    )
    with pytest.raises(error, match=message):
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
