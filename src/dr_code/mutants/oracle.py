"""Execution-derived oracles and behavioral validation gates for mutants.

The oracle mechanism (cf. arXiv:2503.02296): run a candidate program on the
task's existing HumanEval+ test *inputs* in the isolated subprocess, capture
each output, and treat the mutant's outputs as the mutant's new expected
suite. A mutant is accepted only if it clears every gate:

- **terminates** within the timeout on all inputs;
- **deterministic** across two runs (identical outputs);
- **behaviorally distinct** from the canonical solution on >=1 input;
- **serializable/comparable** outputs (``repr`` round-trips finitely).

The subprocess is dr-code's ``run_python_subprocess`` (host isolation, not a
security sandbox). No LLM calls; local compute only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum

from dr_code.humaneval.subprocess_runner import (
    SubprocessError,
    SubprocessRunner,
    run_python_subprocess,
)
from dr_code.models import FrozenModel

# Marker lines delimit the JSON result on stdout so incidental prints in a
# candidate program cannot be mistaken for the protocol payload.
_RESULT_BEGIN = "<<<DR_MUTANT_ORACLE_BEGIN>>>"
_RESULT_END = "<<<DR_MUTANT_ORACLE_END>>>"


class OutcomeKind(StrEnum):
    """Per-input execution outcome kind."""

    VALUE = "value"
    ERROR = "error"


class InputOutcome(FrozenModel):
    """One input's outcome: a ``repr`` value or a raised-exception summary."""

    kind: OutcomeKind
    output_repr: str


class ProgramOutcomes(FrozenModel):
    """Ordered per-input outcomes for one program over one input list."""

    outcomes: tuple[InputOutcome, ...]


class OracleError(RuntimeError):
    """The oracle subprocess broke its protocol or exceeded a limit."""


@dataclass(frozen=True, slots=True)
class GateReport:
    """Why a mutant was accepted or rejected, for the search record."""

    terminates: bool
    deterministic: bool
    distinct_input_count: int
    serializable: bool

    @property
    def accepted(self) -> bool:
        return (
            self.terminates
            and self.deterministic
            and self.serializable
            and self.distinct_input_count >= 1
        )

    def rejection_reason(self) -> str | None:
        if not self.terminates:
            return "did not terminate within timeout on all inputs"
        if not self.serializable:
            return "produced non-serializable output"
        if not self.deterministic:
            return "non-deterministic across two runs"
        if self.distinct_input_count < 1:
            return "behaviorally identical to canonical on every input"
        return None


def _runner_source(entry_point: str) -> str:
    # The child receives {"program": str, "inputs": [[...args...], ...]} and
    # emits a JSON list of per-input outcomes between the markers. Each input
    # is applied as ``entry_point(*args)``. Exceptions are captured as a
    # typed outcome rather than crashing the batch, so a mutant that raises on
    # some inputs is still deterministically comparable.
    return f"""
import json, sys

def _main():
    payload = json.loads(sys.stdin.read())
    namespace = {{}}
    exec(payload["program"], namespace)
    func = namespace[{entry_point!r}]
    outcomes = []
    for args in payload["inputs"]:
        try:
            value = func(*args)
            outcomes.append({{"kind": "value", "output_repr": repr(value)}})
        except Exception as exc:  # noqa: BLE001 - captured as an outcome
            outcomes.append(
                {{"kind": "error", "output_repr": type(exc).__name__}}
            )
    sys.stdout.write({_RESULT_BEGIN!r})
    sys.stdout.write(json.dumps(outcomes))
    sys.stdout.write({_RESULT_END!r})

_main()
"""


def run_program_on_inputs(
    *,
    program: str,
    entry_point: str,
    inputs: tuple[tuple[object, ...], ...],
    timeout_seconds: float,
    run_in_subprocess: SubprocessRunner = run_python_subprocess,
) -> ProgramOutcomes:
    """Execute ``program`` on each input tuple; capture ordered outcomes.

    Raises :class:`OracleError` on subprocess breakage (timeout, output
    overflow, protocol violation) so the caller can record a non-terminating
    or otherwise unusable mutant rather than silently accept partial output.
    """

    input_json = json.dumps(
        {"program": program, "inputs": [list(args) for args in inputs]}
    )
    try:
        completed = run_in_subprocess(
            source=_runner_source(entry_point),
            input_json=input_json,
            timeout_seconds=timeout_seconds,
        )
    except SubprocessError as exc:
        raise OracleError(str(exc)) from exc

    if completed.returncode != 0:
        raise OracleError(
            f"oracle child exited {completed.returncode}: "
            f"{completed.stderr[:200]}"
        )
    return _parse_outcomes(completed.stdout, expected=len(inputs))


def _parse_outcomes(stdout: str, *, expected: int) -> ProgramOutcomes:
    begin = stdout.find(_RESULT_BEGIN)
    end = stdout.find(_RESULT_END)
    if begin < 0 or end < 0 or end < begin:
        raise OracleError("oracle child did not emit a delimited result")
    raw = stdout[begin + len(_RESULT_BEGIN) : end]
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OracleError("oracle result was not valid JSON") from exc
    if not isinstance(decoded, list) or len(decoded) != expected:
        got = len(decoded) if isinstance(decoded, list) else 0
        raise OracleError(
            f"oracle returned {got} outcomes, expected {expected}"
        )
    outcomes = tuple(
        InputOutcome(
            kind=OutcomeKind(item["kind"]), output_repr=item["output_repr"]
        )
        for item in decoded
    )
    return ProgramOutcomes(outcomes=outcomes)


def distinct_input_indices(
    canonical: ProgramOutcomes, mutant: ProgramOutcomes
) -> tuple[int, ...]:
    """Indices where the mutant's outcome differs from the canonical's."""

    if len(canonical.outcomes) != len(mutant.outcomes):
        raise OracleError("outcome lengths differ; inputs were not aligned")
    return tuple(
        index
        for index, (c, m) in enumerate(
            zip(canonical.outcomes, mutant.outcomes, strict=True)
        )
        if c != m
    )


def evaluate_gates(
    *,
    canonical: ProgramOutcomes,
    mutant_first: ProgramOutcomes,
    mutant_second: ProgramOutcomes,
) -> GateReport:
    """Compute the gate report from canonical + two mutant runs.

    ``mutant_first``/``mutant_second`` are two independent executions of the
    same mutant used to check determinism. Termination and serializability are
    implied by having complete, parsed outcomes for both mutant runs (a
    non-terminating or non-serializable run raises before reaching here); the
    caller passes ``terminates``/``serializable`` via successful parsing.
    """

    deterministic = mutant_first == mutant_second
    distinct = distinct_input_indices(canonical, mutant_first)
    return GateReport(
        terminates=True,
        deterministic=deterministic,
        distinct_input_count=len(distinct),
        serializable=True,
    )


__all__ = [
    "GateReport",
    "InputOutcome",
    "OracleError",
    "OutcomeKind",
    "ProgramOutcomes",
    "distinct_input_indices",
    "evaluate_gates",
    "run_program_on_inputs",
]
