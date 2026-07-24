"""Typed execution oracle for behavioral mutants."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from dr_code.execution.subprocess import (
    PythonSubprocessRunner,
    SubprocessError,
    run_python_subprocess,
)
from dr_code.mutants.outcomes import ExecutionOutcome

_RESULT_BEGIN: Final = "<<<DR_CODE_MUTANTS_V2_BEGIN>>>"
_RESULT_END: Final = "<<<DR_CODE_MUTANTS_V2_END>>>"
_PROTOCOL_VERSION: Final = 2


@dataclass(frozen=True, slots=True)
class ProgramOutcomes:
    """Ordered outcomes aligned to one input sequence."""

    outcomes: tuple[ExecutionOutcome, ...]


class OracleError(RuntimeError):
    """The oracle could not produce a complete trustworthy result."""


class OracleInputError(OracleError):
    """Oracle input could not cross the subprocess protocol."""


class OracleExecutionError(OracleError):
    """The subprocess did not complete normally."""


class OracleProtocolError(OracleError):
    """The subprocess response violated the oracle protocol."""


class _WireEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    protocol_version: Literal[2]
    invocation_id: str
    outcomes: list[ExecutionOutcome]


@dataclass(frozen=True, slots=True)
class GateReport:
    """Determinism and divergence gate results."""

    canonical_deterministic: bool
    mutant_deterministic: bool
    distinct_input_count: int

    @property
    def accepted(self) -> bool:
        return (
            self.canonical_deterministic
            and self.mutant_deterministic
            and self.distinct_input_count >= 1
        )

    def rejection_reason(self) -> str | None:
        if not self.canonical_deterministic:
            return "canonical is non-deterministic across two runs"
        if not self.mutant_deterministic:
            return "mutant is non-deterministic across two runs"
        if self.distinct_input_count < 1:
            return "behaviorally identical to canonical on every input"
        return None


_RUNNER_SOURCE: Final = f"""
import ast
import json
import os
import sys

def _main():
    payload = json.loads(sys.stdin.read())
    trusted_fd = os.dup(1)
    try:
        with open(os.devnull, "w") as discarded:
            os.dup2(discarded.fileno(), 1)
            namespace = {{"__name__": "__dr_code_mutant__"}}
            exec(payload["program"], namespace)
            function = namespace[payload["entry_point"]]
            outcomes = []
            for input_repr in payload["input_reprs"]:
                args = ast.literal_eval(input_repr)
                try:
                    value = function(*args)
                except Exception as exc:
                    outcomes.append(
                        {{
                            "kind": "error",
                            "exception_type": (
                                type(exc).__module__
                                + "."
                                + type(exc).__qualname__
                            ),
                            "exception_args_repr": repr(exc.args),
                        }}
                    )
                else:
                    outcomes.append(
                        {{"kind": "value", "value_repr": repr(value)}}
                    )
        envelope = {{
            "invocation_id": payload["invocation_id"],
            "protocol_version": {_PROTOCOL_VERSION},
            "outcomes": outcomes,
        }}
        encoded = (
            {_RESULT_BEGIN!r}
            + json.dumps(envelope, sort_keys=True)
            + {_RESULT_END!r}
        ).encode()
        os.write(trusted_fd, encoded)
    finally:
        os.close(trusted_fd)

_main()
"""


def oracle_runner_source() -> str:
    """Return the current source value for one explicit caller capture."""

    return _RUNNER_SOURCE


def run_program_on_inputs(
    *,
    program: str,
    entry_point: str,
    input_reprs: tuple[str, ...],
    timeout_seconds: float,
    runner: PythonSubprocessRunner = run_python_subprocess,
    runner_source: str | None = None,
) -> ProgramOutcomes:
    """Run one program on literal argument tuples in a fresh process."""

    invocation_id = secrets.token_hex(32)
    try:
        input_text = json.dumps(
            {
                "entry_point": entry_point,
                "input_reprs": input_reprs,
                "invocation_id": invocation_id,
                "program": program,
            },
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise OracleInputError(
            "oracle input is not JSON serializable"
        ) from exc
    try:
        completed = runner(
            source=_RUNNER_SOURCE if runner_source is None else runner_source,
            input_text=input_text,
            timeout_seconds=timeout_seconds,
        )
    except SubprocessError as exc:
        raise OracleExecutionError(str(exc)) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip().replace("\n", " ")[:200]
        raise OracleExecutionError(
            f"oracle child exited {completed.returncode}"
            + (f": {detail}" if detail else "")
        )
    return _parse_outcomes(
        completed.stdout,
        expected_count=len(input_reprs),
        expected_invocation_id=invocation_id,
    )


def _parse_outcomes(
    stdout: str,
    *,
    expected_count: int,
    expected_invocation_id: str,
) -> ProgramOutcomes:
    if not stdout.startswith(_RESULT_BEGIN) or not stdout.endswith(
        _RESULT_END
    ):
        raise OracleProtocolError(
            "oracle did not emit one complete final envelope"
        )
    payload_start = len(_RESULT_BEGIN)
    end = len(stdout) - len(_RESULT_END)
    try:
        envelope = _WireEnvelope.model_validate_json(stdout[payload_start:end])
    except ValidationError as exc:
        raise OracleProtocolError(
            "oracle result did not match protocol version 2"
        ) from exc
    if not secrets.compare_digest(
        envelope.invocation_id,
        expected_invocation_id,
    ):
        raise OracleProtocolError("oracle invocation binding mismatch")
    if len(envelope.outcomes) != expected_count:
        raise OracleProtocolError(
            f"oracle returned {len(envelope.outcomes)} outcomes; "
            f"expected {expected_count}"
        )
    return ProgramOutcomes(outcomes=tuple(envelope.outcomes))


def distinct_input_indices(
    canonical: ProgramOutcomes,
    mutant: ProgramOutcomes,
) -> tuple[int, ...]:
    """Return aligned input indices whose outcomes differ."""

    if len(canonical.outcomes) != len(mutant.outcomes):
        raise OracleProtocolError("outcome sequences are not aligned")
    return tuple(
        index
        for index, (expected, observed) in enumerate(
            zip(canonical.outcomes, mutant.outcomes, strict=True)
        )
        if expected != observed
    )


def evaluate_gates(
    *,
    canonical_first: ProgramOutcomes,
    canonical_second: ProgramOutcomes,
    mutant_first: ProgramOutcomes,
    mutant_second: ProgramOutcomes,
) -> GateReport:
    """Accept only deterministic canonical and mutant divergence."""

    if len(canonical_first.outcomes) != len(canonical_second.outcomes):
        raise OracleProtocolError(
            "canonical outcome sequences are not aligned"
        )
    if len(mutant_first.outcomes) != len(mutant_second.outcomes):
        raise OracleProtocolError("mutant outcome sequences are not aligned")
    distinct = distinct_input_indices(canonical_first, mutant_first)
    return GateReport(
        canonical_deterministic=canonical_first == canonical_second,
        mutant_deterministic=mutant_first == mutant_second,
        distinct_input_count=len(distinct),
    )


__all__ = (
    "GateReport",
    "OracleError",
    "OracleExecutionError",
    "OracleInputError",
    "OracleProtocolError",
    "ProgramOutcomes",
    "distinct_input_indices",
    "evaluate_gates",
    "oracle_runner_source",
    "run_program_on_inputs",
)
